# loss functions for training
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import math

class LossFunction:
    def __init__(self, sigma_1=math.sqrt(1/2), sigma_2=None):
        self.sigma_1 = sigma_1 # the larger the sigma, the less weight on the distance loss
        self.sigma_2 = sigma_2
    
    def __call__(self, dist_ins, dist_target, class_ins, class_target):
        return self.loss(dist_ins, dist_target, class_ins, class_target)
    
    def loss(self, dist_ins, dist_target, class_ins, class_target):
        # only sum the mse loss over the entries where class_target does not point to itself for sigma_1
        mask = class_target != torch.arange(class_target.size(1), device=class_target.device)
        if mask.sum() == 0: # deal with the case where there are no non-self edges, to not produce nan result 
            dist_loss = torch.tensor(0.0, device=dist_ins.device)
            if self.sigma_2 is not None:
                dist_loss += F.mse_loss(dist_ins[~mask], dist_target[~mask]) / (2*self.sigma_2 ** 2)
        else:
            dist_loss = F.mse_loss(dist_ins[mask], dist_target[mask]) / (2*self.sigma_1 ** 2)
            if self.sigma_2 is not None and (~mask).sum() > 0:
                dist_loss += F.mse_loss(dist_ins[~mask], dist_target[~mask]) / (2*self.sigma_2 ** 2)
        class_loss = CrossEntropyLoss()(class_ins, class_target)
        return dist_loss, class_loss
    
    