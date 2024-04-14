
import os
from typing import IO, BinaryIO, Iterable, Optional, Type, Callable, List, Tuple

import numpy.typing as npt
import torch
import math
import numpy as np

import torch
import torch.nn.functional as F

###############################################################

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8):
        
        self.betas = betas
        self.eps = eps
        self.lamda = weight_decay
        self.params = params
        
        super().__init__(params, {"lr": lr})

    ##########################################

    def step(self, closure: Optional[Callable] = None):
        
        for group in self.param_groups:

            for param in group["params"]:
                
                if param.grad is not None:

                    if len(self.state[param]) == 0:
                        self.state[param]['m'] = torch.zeros_like(param.data)
                        self.state[param]['v'] = torch.zeros_like(param.data)

                    ###########################

                    beta_1, beta_2 = self.betas

                    ###########################
                    
                    v = (beta_2 * self.state[param]['v']) + ((1 - beta_2) * (param.grad.data**2))
                    m = ((1 - beta_1) * param.grad.data)  + (self.state[param]['m'] * beta_1)

                    a_t = group["lr"] * math.sqrt(1 - beta_2 **(self.state[param].get("t", 1)))
                    a_t = a_t / (1 - beta_1 **(self.state[param].get("t", 1)))

                    ###########################
                    
                    a_t_adjust = (a_t * m) / (torch.sqrt(v) +self.eps)
                    param.data -= a_t_adjust
                    param.data -= param.data * (self.lamda * group["lr"])

                    ###########################
                    
                    self.state[param]['m'] = m
                    self.state[param]['v'] = v
                    self.state[param]['t'] = 1 + self.state[param].get("t", 1)

                else:
                    continue

####################################################################################


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if cosine_cycle_iters >= it >= warmup_iters:
        
        final_lr = math.cos(math.pi * ((it - warmup_iters) / (cosine_cycle_iters - warmup_iters)))
        final_lr = 0.5 * (1 + final_lr)
        final_lr = final_lr * (max_learning_rate-min_learning_rate)
        final_lr = min_learning_rate + final_lr
        return final_lr
    
    ######################

    if warmup_iters > it:
        
        return max_learning_rate * (it / warmup_iters)
    
    ######################
    
    return min_learning_rate

########################################################################################
    
def gradient_clipping(parameters, max_l2_norm):

    for param in parameters:

        if param.grad is not None:

            clipped_norm = torch.norm(param.grad, p=2)

            if max_l2_norm < clipped_norm:

                parameters = (max_l2_norm * param.grad) / (1e-6 + clipped_norm)

    ######################

    return parameters