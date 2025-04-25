from numpy import dtype
import torch
from losses import MaskedMSELoss
from models import ResNet6
from tqdm import tqdm
from dataloaders import BlendedMVS
from torch.utils.data import DataLoader
from evaluation import evaluate_model
from argparse import ArgumentParser
import numpy as np
from pathlib import Path

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINTS = Path('checkpoints')
CHECKPOINTS.mkdir(parents=True, exist_ok=True)

def main(data_path: str, subset: float, batch_size: int, model: str, epochs: int, lr: float, optimizer: str, scheduler: str):
   # Model
   if model == 'cnn':
      model = ResNet6().to(DEVICE)
   else:
      error_msg = f"Model {model} is not a valid model name"
      raise ValueError(error_msg)
   criterion = MaskedMSELoss()

   # TODO: use function argument
   optimizer =  torch.optim.AdamW(model.parameters(), lr)

   # Scheduler
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

   train_dataset = BlendedMVS(data_path=data_path, subset=subset, partition='train')
   train_loader = DataLoader(
      dataset=train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn
   )
   print(f"Train dataset: {len(train_dataset)} objects | {len(train_loader)} batches")
   val_dataset = BlendedMVS(data_path=data_path, subset=1, partition='val')
   val_loader = DataLoader(
      dataset=val_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn
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
         save_model(model, optimizer, scheduler, best_valid_loss, valid_metrics, epoch, CHECKPOINTS / 'best_model.pth')
         print("Saved best val model")
      train_losses.append(train_loss)
      val_losses.append(valid_loss)

   save_model(model, optimizer, scheduler, best_valid_loss, valid_metrics, epoch, CHECKPOINTS / 'last_model.pth')
   print("Saved last model")
   
   with (CHECKPOINTS / 'losses.txt').open('w') as f:
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

      with torch.autocast(DEVICE):
         if isinstance(model, ResNet6):
            depth_maps_pred = model(images)
         else:
            depth_maps_pred = model(images, intrinsics, extrinsics)
         loss = criterion(depth_maps_pred, depth_maps, mask)

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

if __name__ == '__main__':
   parser = ArgumentParser()
   parser.add_argument('data_path', type=str)
   parser.add_argument('subset', type=float)
   parser.add_argument('batch_size', type=int)
   parser.add_argument('model', type=str)
   parser.add_argument('epochs', type=int)
   parser.add_argument('lr', type=float)
   parser.add_argument('optimizer', type=str)
   parser.add_argument('scheduler', type=str)
   args = parser.parse_args()
   main(
      data_path=args.data_path, 
      subset=args.subset, 
      batch_size=args.batch_size, 
      model=args.model, 
      epochs=args.epochs, 
      lr=args.lr, 
      optimizer=args.optimizer, 
      scheduler=args.scheduler
   )