import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs are logits (raw outputs from the model before sigmoid)
        # targets are 0 or 1 labels

        # Ensure targets has the same shape as inputs (unsqueeze)
        # inputs: [BATCH_SIZE, 1]
        # targets: [BATCH_SIZE] -> need to be [BATCH_SIZE, 1]
        targets = targets.unsqueeze(1) 
        
        # FIX: Cast targets to float type
        targets = targets.float() # Convert from Long to Float

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate p_t (probability of the true class)
        pt = torch.exp(-BCE_loss) 

        # Calculate modulating factor
        focal_term = (1 - pt)**self.gamma

        # Apply alpha weighting
        alpha_term = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_term * focal_term * BCE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else: # 'none'
            return loss