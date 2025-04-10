import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
import numpy as np

class FocalLoss(nn.Module):
    """Multi-class Focal Loss implementation.
    Assumes input logits and target class indices.
    """
    def __init__(self, 
                 alpha: Optional[Union[float, List[float], torch.Tensor]] = 0.25, 
                 gamma: float = 2.0, 
                 reduction: str = 'mean', 
                 num_classes: Optional[int] = None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes # Store num_classes if provided
        
        self.alpha: Optional[Union[float, torch.Tensor]] = None 
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                if num_classes is not None:
                    if not isinstance(num_classes, (int, np.integer)) or int(num_classes) <= 0:
                         raise ValueError(f"num_classes ({num_classes}) must be a positive integer when using scalar alpha.")
                    self.alpha = torch.full((int(num_classes),), float(alpha), dtype=torch.float)
                else:
                     print("[WARN] FocalLoss initialized with scalar alpha but no num_classes. Alpha will not be per-class.")
                     self.alpha = float(alpha) 
            elif isinstance(alpha, (list, torch.Tensor)):
                if num_classes is None:
                    raise ValueError("num_classes must be provided when alpha is a list or tensor.")
                alpha_tensor = torch.tensor(alpha) if isinstance(alpha, list) else alpha
                if not isinstance(num_classes, int) or len(alpha_tensor) != num_classes:
                     raise ValueError(f"Length of alpha ({len(alpha_tensor)}) must match num_classes ({num_classes}) if alpha is a list/tensor.")
                self.alpha = alpha_tensor.float()
            else:
                 raise TypeError("alpha must be float, int, list, torch.Tensor, or None.")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args:
            inputs (torch.Tensor): Logits of shape [batch_size, num_classes].
            targets (torch.Tensor): Ground truth class indices of shape [batch_size].
        """
        if not torch.is_tensor(inputs) or not torch.is_tensor(targets):
            raise TypeError("Inputs and targets must be torch.Tensor")
        if inputs.ndim != 2:
            raise ValueError(f"Inputs must have 2 dimensions ([batch_size, num_classes]), got {inputs.ndim}")
        if targets.ndim != 1:
            raise ValueError(f"Targets must have 1 dimension ([batch_size]), got {targets.ndim}")
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError(f"Input batch size ({inputs.shape[0]}) must match target batch size ({targets.shape[0]}).")

        num_classes_runtime = inputs.shape[1]
        if self.num_classes is not None and self.num_classes != num_classes_runtime:
             raise ValueError(f"Number of classes in input ({num_classes_runtime}) does not match initialized num_classes ({self.num_classes})")

        targets = targets.long()
        log_softmax_inputs = F.log_softmax(inputs, dim=1)
        
        # Clamp targets *before* nll_loss to prevent CUDA errors
        targets_clamped = torch.clamp(targets, 0, num_classes_runtime - 1)
        if not torch.equal(targets, targets_clamped):
             print(f"[WARN] FocalLoss: Targets contained values outside [0, {num_classes_runtime - 1}]. Clamping applied.")
             
        nll_loss = F.nll_loss(log_softmax_inputs, targets_clamped, reduction='none')
        pt = torch.exp(-nll_loss)
        focal_term = (1 - pt)**self.gamma
        focal_loss = focal_term * nll_loss

        if self.alpha is not None:
            current_alpha = self.alpha
            if isinstance(current_alpha, torch.Tensor):
                if current_alpha.ndim == 0: 
                     alpha_t = current_alpha 
                elif current_alpha.ndim == 1:
                     if current_alpha.device != inputs.device:
                          current_alpha = current_alpha.to(inputs.device)
                     if len(current_alpha) != num_classes_runtime:
                          raise ValueError(f"Initialized alpha tensor length ({len(current_alpha)}) does not match input num_classes ({num_classes_runtime})")
                     alpha_t = current_alpha.gather(0, targets_clamped.data.view(-1))
                     focal_loss = alpha_t * focal_loss
                else:
                     raise ValueError("alpha tensor must be 1D or scalar.")
            elif isinstance(current_alpha, float):
                 focal_loss = current_alpha * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'none':
             return focal_loss
        else:
            raise ValueError(f"Invalid reduction type: {self.reduction}. Choose 'mean', 'sum', or 'none'.") 