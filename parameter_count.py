import torch
from models import ProjeXion
from torchinfo import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'
images = torch.rand((32, 1 + 5, 3, 160, 160), device=device)
intrinsics = torch.rand((32, 1 + 5, 3, 3), device=device)
extrinsics = torch.rand((32, 1 + 5, 4, 4), device=device)
model = ProjeXion(n_depths=25).to(device)
model_summary = summary(model, input_data=[images, intrinsics, extrinsics])
print(str(model_summary))