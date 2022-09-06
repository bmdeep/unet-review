import torch


EPSILON = 1e-6

class DiceLoss(torch.nn.Module):
    def __init__(self,):
        super().__init__()
    
    def forward(self, pred, mask):
        pred = pred.flatten()
        mask = mask.flatten()
        
        intersect = (mask * pred).sum()
        dice_score = 2*intersect / (pred.sum() + mask.sum() + EPSILON)
        dice_loss = 1 - dice_score
        return dice_loss
