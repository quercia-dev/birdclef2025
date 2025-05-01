import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', task_type='multi-class', num_classes=None):
        """
        Numerically stable Focal Loss implementation for multi-class and multi-label classification.
        
        :param gamma: Focusing parameter (default: 2)
        :param alpha: Balancing factor, can be scalar or tensor for class-wise weights
        :param reduction: Specifies reduction method: 'none'|'mean'|'sum'
        :param task_type: Specifies task type: 'multi-class' or 'multi-label'
        :param num_classes: Number of classes (required for multi-class)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes
        self.eps = 1e-7  # Epsilon for numerical stability

        # Handle alpha for class balancing
        if task_type == 'multi-class' and alpha is not None and isinstance(alpha, (list, torch.Tensor)):
            assert num_classes is not None, "num_classes must be specified for multi-class with alpha"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        if self.task_type == 'multi-class':
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == 'multi-label':
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(f"Unsupported task_type '{self.task_type}'. Use 'multi-class' or 'multi-label'.")

    def multi_class_focal_loss(self, inputs, targets):
        """Numerically stable focal loss for multi-class classification."""
        # Convert one-hot encoded targets to class indices if needed
        if targets.dim() > 1 and targets.shape[1] > 1:
            targets = torch.argmax(targets, dim=1)

        targets = targets.long()
        
        # Apply log_softmax for numerical stability instead of softmax
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Get log probability for the correct class for each sample
        batch_size = inputs.size(0)
        log_p_t = log_probs[torch.arange(batch_size, device=targets.device), targets]
        
        # Convert to probabilities for the focal weighting
        p_t = torch.exp(log_p_t)
        
        # Compute focal weight with gradient clipping for stability
        # Clamp p_t to avoid underflow/overflow in the power operation
        p_t = torch.clamp(p_t, min=self.eps, max=1.0 - self.eps)
        focal_weight = torch.clamp((1 - p_t), min=self.eps, max=1.0) ** self.gamma
        
        # Use NLL loss which works with log_softmax outputs for stability
        # This is equivalent to cross entropy but more stable
        loss = -focal_weight * log_p_t
        
        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            # Ensure alpha is on the same device as targets
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss
        
        # Handle NaN values safely
        loss = torch.where(torch.isnan(loss), torch.full_like(loss, 0.0), loss)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_label_focal_loss(self, inputs, targets):
        """Numerically stable focal loss for multi-label classification."""
        # For numerical stability, use logits and BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Get probabilities for the focal weight calculation
        probs = torch.sigmoid(inputs)
        
        # Compute p_t and focal weight with clamping for stability
        p_t = probs * targets + (1 - probs) * (1 - targets)
        p_t = torch.clamp(p_t, min=self.eps, max=1.0 - self.eps)
        focal_weight = torch.clamp((1 - p_t), min=self.eps, max=1.0) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weight
        loss = focal_weight * bce_loss
        
        # Handle NaN values safely
        loss = torch.where(torch.isnan(loss), torch.full_like(loss, 0.0), loss)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss