import torch
import math

class CosineAnnealingWithWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self, 
        optimizer, 
        decay_epoch, 
        warmup_epoch, 
        eta_min_ratio = 0.001, 
        last_epoch = -1
    ):
        self.decay_epoch = decay_epoch
        self.warmup_epoch = warmup_epoch
        self.eta_min_ratio = eta_min_ratio
        
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epoch:
            multiplier = (self.last_epoch + 1) / self.warmup_epoch
        else:
            T_cur = self.last_epoch - self.warmup_epoch
            T_total = self.decay_epoch - self.warmup_epoch
            
            if T_cur >= T_total:
                multiplier = self.eta_min_ratio
            else:
                cosine_decay = 0.5 * (1 + math.cos(math.pi * T_cur / T_total))
                multiplier = self.eta_min_ratio + (1 - self.eta_min_ratio) * cosine_decay

        return [base_lr * multiplier for base_lr in self.base_lrs]