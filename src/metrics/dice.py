import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleclassDiceLoss(nn.Module):
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
    

class MulticlassDiceLoss(nn.Module):
    def __init__(self, num_classes=1, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)

        if targets.ndim == 3:
            targets = F.one_hot(targets, num_classes=self.num_classes + 1).permute(0, 3, 1, 2).float()

        probs = probs[:, 1:, :, :]
        targets = targets[:, 1:, :, :]

        intersection = torch.sum(probs * targets, dim=(0, 2))
        cardinality = torch.sum(probs + targets, dim=(0, 2))

        dice_score = (2. * intersection + self.eps) / (cardinality + self.eps)
        
        return 1 - torch.mean(dice_score)

class DiceLoss(nn.Module):
    def __init__(self, num_classes=55, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits, targets):
        channels = logits.shape[1]

        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)
            targets = targets.view(probs.shape).float()
        else:
            probs = F.softmax(logits, dim=1)
            
            if targets.ndim == 3: # (B, H, W)
                targets = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
            
            probs = probs[:, 1:, :, :]
            targets = targets[:, 1:, :, :]

        dims = (0, 2, 3) 
        intersection = torch.sum(probs * targets, dim=dims)
        cardinality = torch.sum(probs + targets, dim=dims)

        dice_score = (2. * intersection + self.eps) / (cardinality + self.eps)
        
        return 1 - torch.mean(dice_score)