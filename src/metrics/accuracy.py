import torch
import torch.nn as nn

def accuracy(y_hat, y):
   return torch.mean((torch.argmax(y_hat, dim=1) == y).float())

class Accuracy(nn.Module):
   def __init__(self):
       super().__init__()

   def forward(self, y_hat, y):
      return torch.mean((torch.argmax(y_hat, dim=1) == y).float())
