import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        probs = probs.view(probs.shape[0], -1)
        targets = targets.view(targets.shape[0], -1).float()

        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2. * intersection + self.smooth) / (denominator + self.smooth)
        
        return 1 - dice.mean()