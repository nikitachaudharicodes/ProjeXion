from multiprocessing import context
import torch
from losses import MaskedMSELoss
from losses.masked_cauchy_loss import MaskedCauchyLoss
from models import ResNet6, MVSNet
from tqdm import tqdm
from dataloaders import BlendedMVS
from torch.utils.data import DataLoader
from evaluation import evaluate_model
from argparse import ArgumentParser
import numpy as np
from pathlib import Path
from torchvision.transforms.v2 import Resize
import json

from models.projeXion import ProjeXion

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(
      data_path: str,
      subset: float,
      context_size: int,
      n_depths: int,
      batch_size: int,
      architecture: str,
      loss: str,
      epochs: int,
      lr: float,
      optimizer: str,
      scheduler: str,
      run_name: str,
      img_height: int = 160,
      img_width: int = 160,
   ):

   checkpoint_path = Path('checkpoints', run_name)
   checkpoint_path.mkdir(parents=True, exist_ok=True)

   # Model
   if architecture == 'cnn':
      model = ResNet6().to(DEVICE)
   elif architecture == 'mvsnet':
      model = MVSNet(n_depths).to(DEVICE)
   elif architecture == 'projexion':
      model = ProjeXion(n_depths).to(DEVICE)
   else:
      error_msg = f"Model {architecture} is not a valid model name"
      raise ValueError(error_msg)
   
   if loss == 'cauchy':
      criterion = MaskedCauchyLoss(c=100)
   else:   
      criterion = MaskedMSELoss()

   # TODO: use function argument
   optimizer_name = optimizer
   optimizer =  torch.optim.AdamW(model.parameters(), lr)

   # Scheduler
   scheduler_name = scheduler
   if scheduler == 'ReduceLROnPlateu':
      scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
         optimizer=optimizer, mode='min', factor=0.1, patience=1
      )
   elif scheduler == 'ExponentialLR':
      scheduler = torch.optim.lr_scheduler.ExponentialLR(
         optimizer=optimizer, gamma=0.8
      )
   else:
      scheduler = torch.optim.lr_scheduler.ConstantLR(
         optimizer=optimizer, factor=1
      )
   scaler = torch.GradScaler(DEVICE)

   # ==============================================================================================
   # Data sets
   # ==============================================================================================
   # Train
   train_dataset = BlendedMVS(
      data_path=data_path, subset=subset, partition='train', context_size=context_size,
      height=img_height, width=img_width
   )
   train_loader = DataLoader(
      dataset=train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn, num_workers=2
   )
   print(f"Train dataset: {len(train_dataset)} objects | {len(train_loader)} batches")
   # Validation
   val_dataset = BlendedMVS(
      data_path=data_path, subset=1, partition='val',
      height=img_height, width=img_width
   )
   val_loader = DataLoader(
      dataset=val_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn, num_workers=2
   )
   print(f"Validation dataset: {len(val_dataset)} objects | {len(val_loader)} batches")

   # TODO: Add wandb to restart training
   last_epoch_completed = 0
   best_valid_loss = float("inf")

   train_losses = []
   val_losses = []
   for epoch in range(last_epoch_completed, epochs):

      print("\nEpoch: {}/{}".format(epoch + 1, epochs))

      curr_lr = scheduler.get_last_lr()[0]

      train_loss = train_model(model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer, scaler=scaler)
      valid_loss, valid_metrics = evaluate_model(model=model, val_loader=val_loader, criterion=criterion)
      
      if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
         scheduler.step(valid_loss)
      else:
         scheduler.step()

      print("\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_loss, curr_lr))
      print("\tVal Loss {:.04f}\t Absolute Relative Error {:.04f}".format(valid_loss, valid_metrics['Abs Rel']))

      if valid_loss <= best_valid_loss:
         best_valid_loss = valid_loss
         save_model(model, optimizer, scheduler, best_valid_loss, valid_metrics, epoch, checkpoint_path / 'best_model.pth')
         print("Saved best val model")
      train_losses.append(train_loss)
      val_losses.append(valid_loss)

   save_model(model, optimizer, scheduler, best_valid_loss, valid_metrics, epoch, checkpoint_path / 'last_model.pth')
   config = {
      'data_path': data_path, 
      'subset': subset,
      'context_size': context_size,
      'n_depths': n_depths,
      'batch_size': batch_size, 
      'model': architecture,
      'loss': loss,
      'epochs': epochs, 
      'lr': lr, 
      'optimizer': optimizer_name,
      'scheduler': scheduler_name,
      'run_name': run_name,
   }
   save_config(config, path=checkpoint_path / 'config.json')
   save_metrics(metrics=valid_metrics, path=checkpoint_path / 'metrics.json')
   print("Saved last model")
   
   with (checkpoint_path / 'losses.txt').open('w') as f:
      f.write('train,valid\n')
      f.writelines([f'{train_loss:.4f},{val_loss:.4f}\n' for train_loss, val_loss in zip(train_losses, val_losses)])


def train_model(model, train_loader, criterion, optimizer, scaler):
   model.train()
   batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Training')

   total_loss = 0

   for i, data in enumerate(train_loader):
      optimizer.zero_grad()
      data = map(lambda x: x.to(DEVICE), data)
      images, intrinsics, extrinsics, depth_maps = data
      mask = depth_maps > 0

      H, W = depth_maps.shape[-2:]
      pred_to_target_size = Resize((H, W))


      with torch.autocast(DEVICE):
         if isinstance(model, ResNet6):
            pred_depths = model(images)
            pred_depths = pred_to_target_size(pred_depths)
            loss = criterion(pred_depths, depth_maps, mask)
         else:
            initial_depth_map_pred, refined_depth_map_pred = model(images, intrinsics, extrinsics)
            initial_depth_map_pred = pred_to_target_size(initial_depth_map_pred)
            refined_depth_map_pred = pred_to_target_size(refined_depth_map_pred)
            loss = criterion(initial_depth_map_pred, depth_maps, mask) + criterion(refined_depth_map_pred, depth_maps, mask)

      total_loss += loss.item()

      batch_bar.set_postfix(
         loss="{:.04f}".format(float(total_loss / (i + 1))),
         lr="{:.06f}".format(float(optimizer.param_groups[0]['lr'])))

      batch_bar.update() # Update tqdm bar

      scaler.scale(loss).backward() # This is a replacement for loss.backward()
      scaler.step(optimizer) # This is a replacement for optimizer.step()
      scaler.update() # This is something added just for FP16

      del images, intrinsics, extrinsics, depth_maps, data, mask, loss
      torch.cuda.empty_cache()

   batch_bar.close() # You need this to close the tqdm bar

   return total_loss / len(train_loader)

def save_model(model, optimizer, scheduler, valid_loss, metrics, epoch, path: Path):
   torch.save(
      {
         'model_state_dict'         : model.state_dict(),
         'optimizer_state_dict'     : optimizer.state_dict() if optimizer is not None else {},
         'scheduler_state_dict'     : scheduler.state_dict() if scheduler is not None else {},
         'valid_loss'               : valid_loss,
         'metrics'                  : metrics,
         'epoch'                    : epoch,
      },
      path
   )

def load_config(path: str):
   return json.load(path)

def save_config(config: dict, path: str):
   with open(path, 'w') as file:
      json.dump(config, file)

def save_metrics(metrics: dict, path: str):
   with open(path, 'w') as file:
      json.dump(metrics, file)

if __name__ == '__main__':
   parser = ArgumentParser()
   parser.add_argument('data_path', type=str)
   parser.add_argument('subset', type=float)
   parser.add_argument('context_size', type=int)
   parser.add_argument('n_depths', type=int)
   parser.add_argument('batch_size', type=int)
   parser.add_argument('model', type=str)
   parser.add_argument('loss', type=str)
   parser.add_argument('epochs', type=int)
   parser.add_argument('lr', type=float)
   parser.add_argument('optimizer', type=str)
   parser.add_argument('scheduler', type=str)
   parser.add_argument('run_name', type=str)
   args = parser.parse_args()
   main(
      data_path=args.data_path, 
      subset=args.subset,
      context_size=args.context_size,
      n_depths=args.n_depths,
      batch_size=args.batch_size, 
      architecture=args.model,
      loss=args.loss,
      epochs=args.epochs, 
      lr=args.lr, 
      optimizer=args.optimizer, 
      scheduler=args.scheduler,
      run_name=args.run_name,
   )