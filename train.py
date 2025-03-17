import torch
from models import ResNet6
from tqdm import tqdm
from dataloaders import BlendedMVS
from torch.utils.data import DataLoader
from evaluation import evaluate_model
from argparse import ArgumentParser

def main(data_path: str, subset: float, batch_size: int, model: str, epochs: int, lr: float, optimizer: str, scheduler: str):
   # Model
   if model == 'cnn':
      model = ResNet6()
   else:
      error_msg = f"Model {model} is not a valid model name"
      raise ValueError(error_msg)
   criterion = torch.nn.MSELoss()

   # TODO: use function argument
   optimizer =  torch.optim.AdamW(model.parameters(), lr)

   # Scheduler
   if scheduler == 'ReduceLROnPlateu':
      scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
         optimizer=optimizer, mode='min', factor=0.1, patience=1
      )
   elif scheduler == 'ReduceLROnPlateu':
      scheduler = torch.optim.lr_scheduler.ExponentialLR(
         optimizer=optimizer, gamma=0.8
      )
   else:
      scheduler = torch.optim.lr_scheduler.ConstantLR(
         optimizer=optimizer, factor=1
      )
   scaler = torch.GradScaler(DEVICE)

   train_dataset = BlendedMVS(data_path=data_path, subset=subset)
   train_loader = DataLoader(
      dataset=train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn
   )

   # TODO: Add wandb to restart training
   last_epoch_completed = 0
   best_lev_dist = float("inf")

   for epoch in range(last_epoch_completed, epochs):

      print("\nEpoch: {}/{}".format(epoch + 1, epochs))

      curr_lr = scheduler.get_last_lr()[0]

      train_loss = train_model(model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer, scaler=scaler)
      # TODO: Replace the line below
      # valid_loss, valid_metrics = evaluate_model(model=model, val_loader=val_loader)
      valid_loss, valid_metrics = 0, None
      
      if scheduler == 'ReduceLROnPlateau':
         scheduler.step(valid_loss)
      else:
         scheduler.step()

      print("\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_loss, curr_lr))
      # print("\tVal Loss {:.04f}".format(valid_loss)) # TODO: print metrics

      # if valid_dist <= best_lev_dist:
      #    best_lev_dist = valid_dist
      #    save_model(model, optimizer, scheduler, ['valid_dist', valid_dist], epoch, best_model_path)
      #    print("Saved best val model")
   save_model(model, optimizer, scheduler, valid_metrics, epoch, 'checkpoints')
   print("Saved last model")

def train_model(model, train_loader, criterion, optimizer, scaler):
   model.train()
   batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Training')

   total_loss = 0

   for i, data in enumerate(train_loader):
      optimizer.zero_grad()

      x, y, lx, ly = data
      x, y = x.to(device), y.to(device)
      lx, ly = lx.to(device), ly.to(device)

      with torch.autocast(device):
         h, lh = model(x, lx)
         h = torch.permute(h, (1, 0, 2))
         loss = criterion(h, y, lh, ly)

      total_loss += loss.item()

      batch_bar.set_postfix(
         loss="{:.04f}".format(float(total_loss / (i + 1))),
         lr="{:.06f}".format(float(optimizer.param_groups[0]['lr'])))

      batch_bar.update() # Update tqdm bar

      scaler.scale(loss).backward() # This is a replacement for loss.backward()
      scaler.step(optimizer) # This is a replacement for optimizer.step()
      scaler.update() # This is something added just for FP16

      del x, y, lx, ly, h, lh, loss
      torch.cuda.empty_cache()

   batch_bar.close() # You need this to close the tqdm bar

   return total_loss / len(train_loader)

def save_model(model, optimizer, scheduler, metrics, epoch, path):
   torch.save(
      {
         'model_state_dict'        : model.state_dict(),
         'optimizer_state_dict'     : optimizer.state_dict() if optimizer is not None else {},
         'scheduler_state_dict'     : scheduler.state_dict() if scheduler is not None else {},
         'metrics'                  : metrics,
         'epoch'                    : epoch,
      },
      path
   )

if __name__ == '__main__':
   DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
   parser = ArgumentParser()
   parser.add_argument('pfm_file')
   args = parser.parse_args()
   main()