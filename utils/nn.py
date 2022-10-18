import torch
import torch.nn as nn

class SelectiveOutput(nn.Linear):
    """
    Linear output transformation: returns vector with output probabilities at target positions, inputs are copied at evidence positions
    """
    def __init__(self, in_features, out_features, class_mask, bias, dtype):
        """
        class_mask: binary matrix of shape (outputs, outputs), with 1 on entry (i,j) when outputs i and j represent classes of the same var (symmetric)
        """
        super(SelectiveOutput, self).__init__(in_features, out_features, bias=bias, dtype=dtype)
        self.class_mask = class_mask

    def __str__(self):
        return f'SelectiveOutput(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias})'

    def forward(self, h, x, mask):
        """
        Input:
        h: (*, in_features) input to the linear transformation
        x: (*, out_features) contains binary outputs at specific locations
        mask: (*, n) integer tensor with 0 for positions needing replacement by corresponding entries in x (evidence), 1 for targets
        softmax applied to scores at positions where mask == 1, grouped per var as indicated by class_mask, else replaced by x entries
        """
        logits = super().forward(h)  # (meaningless predictions for input entries)
        return torch.where(mask == 1, Softmax(logits, self.class_mask), x)

def Softmax(logits, class_mask):
    """
    logits: (*, n) activations for all evidence/target classes
    class_mask: (n, n) binary matrix, with 1 on entry (i, j) when entries i and j in logits tensor represent classes of the same var (symmetric)
    returns: (*, n) representing the class probabilities (transformed logits, normalized per var)
    """
    x = torch.exp(logits) # dim: (*, n)
    denom = torch.matmul(x, class_mask.to(torch.float64)) # dim: (*, n) . (n, n) = (*, n), group classes belonging to same var and take sum per var
    softmax = x/denom # dim (*, n)
    return softmax # logits rescaled as probabilities, with probs of all classes belonging to same var summing to 1

def SelectiveCELoss(pred, mask, target):
    """
    pred: (*, n) predicted probabilities for all evidence/target classes
    mask: (*, n) integer tensor with 0 for evidence positions, 1 for targets
    target: (*, n) binary tensor with observed targets, only entries for target variables (as indicated by mask) are used
    returns: (*,) loss, divided by the number of targets, only calculated for target variables, indicated by 1 in the mask
    """
    loss = - target * torch.log(pred)
    loss = torch.where(mask == 1., loss, 0.)
    num_targets = torch.sum(mask, -1, keepdim=False) 
    num_targets = torch.where(num_targets > 0, num_targets, torch.ones_like(num_targets)) # to avoid nan loss for all-zero mask
    return torch.sum(loss, -1, keepdim=False)/num_targets # weigh total loss according to number of targets

def WeightedMSELoss(pred, target, mask, class_mask):
    """
    pred: (*, n) predicted probabilities for all classes
    target: (*, n) tensor with desired probabilities, only entries for target variables (as indicated by mask) are used
    mask: (*, n) integer tensor with 0 for evidence positions, 1 for targets
    class_mask: (n, n) binary matrix, with 1 on entry (i, j) when entries i and j in pred and target tensors represent classes of the same var (symmetric)
    returns: (*, ) mean square error loss, only calculated for target variables, indicated by 1 in the mask. loss contributions are divided by number of classes per var
             and total error is also divided by the number of target variables in the query
    """
    mse = (pred - target)**2
    mse = torch.where(mask == 1., mse, 0.0)
    mse = torch.matmul(mse, class_mask.to(torch.float64)) # aggregate loss according to classes belonging to the same var
    mse = mse/torch.sum(class_mask, dim=1)**2 # weigh loss according to number of classes per var
    num_targets = torch.sum(mask, -1, keepdim=False) 
    num_targets = torch.where(num_targets > 0, num_targets, torch.ones_like(num_targets)) # to avoid nan loss for all-zero mask
    return torch.sum(mse, -1, keepdim=False)/num_targets # weigh total loss according to number of targets

def WeightedMAE(pred, target, mask, class_mask):
    """
    pred: (*, n) predicted probabilities for all classes
    target: (*, n) tensor with desired targets, only entries for target variables (as indicated by mask) are used
    mask: (*, n) integer tensor with 0 for evidence positions, 1 for targets
    class_mask: (n, n) binary matrix, with 1 on entry (i, j) when entries i and j in pred and target tensors represent classes of the same var (symmetric)
    returns: (*, ) mean absolute error, only calculated for target variables, indicated by 1 in the mask. error contributions are divided by number of classes per var
             and total error is also divided by the number of target variables in the query
    """
    mae = torch.abs(pred - target)
    mae = torch.where(mask == 1., mae, 0.0)
    mae = torch.matmul(mae, class_mask.to(torch.float64)) # aggregate mae according to classes belonging to the same var
    mae = mae/torch.sum(class_mask, dim=1)**2 # weigh mae according to number of classes per var
    num_targets = torch.sum(mask, -1, keepdim=False) 
    num_targets = torch.where(num_targets > 0, num_targets, torch.ones_like(num_targets)) # to avoid nan loss for all-zero mask
    return torch.sum(mae, -1, keepdim=False)/num_targets # weigh total loss according to number of targets