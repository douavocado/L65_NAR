# loss functions for training
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import math

class LossFunction:
    def __init__(self, sigma_1=None, sigma_2=None):
        """ Sigma_1 is the inverse weight given to the distance loss when class_target does not point to itself.
            Sigma_2 is the inverse weight given to the distance loss when class_target points to itself.
        """
        self.sigma_1 = sigma_1 # the larger the sigma, the less weight on the distance loss
        self.sigma_2 = sigma_2
        self.cross_entropy_loss = CrossEntropyLoss()
    
    def __call__(self, dist_ins, dist_target, class_ins, class_target):
        return self.loss(dist_ins, dist_target, class_ins, class_target)
    
    def loss(self, dist_ins, dist_target, class_ins, class_target):
        dist_loss = torch.tensor(0.0, device=dist_ins.device)
        # only sum the mse loss over the entries where class_target does not point to itself for sigma_1
        mask = class_target != torch.arange(class_target.size(1), device=class_target.device)
        if self.sigma_1 is not None and mask.sum() > 0: # deal with the case where there are no non-self edges, to not produce nan result 
            dist_loss = F.mse_loss(dist_ins[mask], dist_target[mask]) / (2*self.sigma_1 ** 2)
        
        # only sum the mse loss over the entries where class_target points to itself for sigma_2
        if self.sigma_2 is not None and (~mask).sum() > 0: # deal with the case where there are all non-self edges, to not produce nan result 
            dist_loss += F.mse_loss(dist_ins[~mask], dist_target[~mask]) / (2*self.sigma_2 ** 2)

        class_loss = self.cross_entropy_loss(class_ins, class_target)
        return dist_loss, class_loss
    
    