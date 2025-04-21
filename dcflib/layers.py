import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from einops import rearrange


## NOTE: sparse coding
from .lasso.linear import dict_learning, sparse_encode
def solve_lasso(X, m, alpha=1e-3):
    """
     Args:
        X: shape [d1, d2], d1 is number of samples
     Output:
        coeffs: shape [d1, m]
        dictionary.t(): shape [m, d2]
    """
    data = X
    device = data.device #"cpu"

    # alpha = 1e-2
    ## constrained: True, constrain the value but low sparsity; False, leads to high sparsity
    dictionary, losses = dict_learning(data, n_components=m, alpha=alpha, constrained=True, steps=10, algorithm='cd', device=device)
    coeffs = sparse_encode(data, dictionary, alpha=alpha, maxiter=20, algorithm='ista', verbose=False) # return shape: [n_samples, n_components]

    return coeffs, dictionary.t()


class DCFConv2d(nn.Module):
    def __init__(
        self, 
        base_layer,
        nlayers: str = "bases_l1", ## options: [bases_l1, bases_l2, coeff_l1, coeff_bases_l1]
        m: int = 9,
        m1: int = 1,
        kc: int = 1,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.requires_grad = False
        if isinstance(base_layer, nn.Conv2d):
            in_channels, out_channels = base_layer.in_channels, base_layer.out_channels
            self.kernel_size = base_layer.kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        k = self.kernel_size
        self.m1 = m1
        self.m = m
        self.sr0 = kc if (self.in_channels // self.base_layer.groups) % kc == 0 else 1 ## ensure the input channel can be divided by kc
        self.sr1 = kc if self.out_channels % kc == 0 else 1 ## ensure the input channel can be divided by kc
        self.nlayers = nlayers
        if nlayers == "bases_l2":
            self.coeff_of_bases = nn.Parameter(
                    torch.ones((1, m * m1 * self.in_channels, 1, 1))
                )
            nn.init.kaiming_normal_(self.coeff_of_bases, math.sqrt(5))
            self.bases_of_bases = nn.Parameter(
                    torch.zeros((m * m1, k[0] * k[1]))
                )
        elif nlayers == "bases_l1":
            self.bases = nn.Parameter(
                torch.zeros((m, k[0], k[1]))
            )
        elif nlayers == "coeff_l1":
            self.bases = nn.Parameter(
                    torch.zeros((m * self.sr0 * self.sr1, k[0] * self.sr0, k[1] * self.sr1))
                )
        elif nlayers == "coeff_bases_l1":
            self.coeff_of_bases = nn.Parameter(
                    torch.ones((1, m * m1 * self.in_channels, 1, 1))
                )
            nn.init.kaiming_normal_(self.coeff_of_bases, math.sqrt(5))
            self.bases_of_bases = nn.Parameter(
                    torch.zeros((m * m1, k[0] * k[1]))
                )
            self.bases_of_coeff = nn.Parameter(
                    torch.zeros((m * self.sr0 * self.sr1, m * self.sr0 * self.sr1))
                )
            nn.init.kaiming_normal_(self.bases_of_coeff, math.sqrt(5))
        else:
            raise NotImplementedError
        
        self.coeff = nn.Parameter(torch.zeros([self.out_channels // self.sr1, self.in_channels // self.sr0, m * self.sr0 * self.sr1]))
        self.init_coeff()
        self.coeff.requires_grad = False

    def init_coeff(self):
        ## use sparse coding to initialize coeff and bases
        data = self.base_layer.state_dict()['weight'] # shape: cout, cin, k1, k2
        data = rearrange(data, "(o p) (i q) k l -> (o i) (p k q l)", p=self.sr1, q=self.sr0)
        coeff, bases = solve_lasso(data, self.m * self.sr0 * self.sr1, alpha=1e-3)
        self.coeff.data = rearrange(coeff, "(o i) m -> o i m", o=self.out_channels // self.sr1)

        # ## DEBUG
        print("||AD - W||: ", torch.dist(data, coeff @ bases).item(), 
              "; ||W||: ", torch.norm(data).item(), 
              "; ||A||: ", torch.norm(coeff).item(),
              "; ||D||: ", torch.norm(bases).item())
    
    def forward(self, x: torch.Tensor):
        base_out = self.base_layer(x)
        
        if self.nlayers == "bases_l2":
            bases = self.bases_of_bases.repeat_interleave(self.in_channels, 0).reshape(1, -1, self.kernel_size[0], self.kernel_size[0])
            bases = bases * self.coeff_of_bases
            b, l, h, w = bases.shape
            coeff = self.coeff
            coeff = coeff.reshape(self.out_channels//self.sr1, self.in_channels//self.sr0, self.m,self.sr1, self.sr0)
            coeff = coeff.permute(0,3, 1,4, 2).reshape(self.out_channels, self.in_channels, self.m)
            bases = bases.reshape(b, self.m1, self.m * self.in_channels, h, w).sum(1).squeeze().reshape(self.in_channels, self.m, h, w)
            weight = torch.einsum('oim, impq-> oipq', coeff, bases)

        elif self.nlayers == "bases_l1":
            weight = torch.einsum('cvm, mki-> cvki', self.coeff.view(self.out_channels, self.in_channels, -1), 
                                self.bases).reshape(self.out_channels, -1, self.kernel_size[0], self.kernel_size[1])
        
        elif self.nlayers == "coeff_l1":
            weight = torch.einsum('cvm, mki-> cvki', self.coeff.view(self.out_channels // self.sr1, self.in_channels // self.sr0, -1), 
                                self.bases).reshape(self.out_channels // self.sr1, self.in_channels // self.sr0, 
                                self.sr1, self.kernel_size[0], self.sr0, self.kernel_size[1]).permute(0, 2, 1, 4, 3, 5).reshape(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        
        elif self.nlayers == "coeff_bases_l1":
            bases = self.bases_of_bases.repeat_interleave(self.in_channels, 0).reshape(1, -1, self.kernel_size[0], self.kernel_size[0])
            bases = bases * self.coeff_of_bases
            b, l, h, w = bases.shape
            bases = bases.reshape(b, self.m1, self.m * self.in_channels, h, w).sum(1).squeeze().reshape(self.m *self.in_channels, 1, h, w)
            coeff = self.coeff
            coeff = coeff.reshape(self.out_channels//self.sr1, self.in_channels//self.sr0, self.m,self.sr1, self.sr0)
            coeff = coeff.permute(0,3, 1,4, 2).reshape(self.out_channels, self.in_channels, self.m)
            bases = bases.reshape(self.in_channels, self.m, h, w)
            weight = torch.einsum('oim, impq-> oipq', coeff, bases)
        else:
            raise NotImplementedError
        
        dcf_out = F.conv2d(
                x,
                weight,
                None, self.base_layer.stride, self.base_layer.padding, self.base_layer.dilation, self.base_layer.groups
            )
        return base_out + dcf_out


class DCFLinear(nn.Module):
    def __init__(
        self, 
        base_layer,
        m: int = 4,
        kc: int = 1,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.requires_grad = False
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        self.in_features = in_features
        self.out_features = out_features
        
        self.m = m
        self.sr0 = kc if self.in_features % kc == 0 else 1 ## ensure the input channel can be divided by kc
        self.sr1 = kc if self.out_features % kc == 0 else 1 ## ensure the input channel can be divided by kc
        self.bases = nn.Parameter(
                    torch.zeros((m * self.sr0 * self.sr1, self.sr0, self.sr1))
                )
        self.coeff = nn.Parameter(torch.zeros([self.out_features // self.sr1, self.in_features // self.sr0, m * self.sr0 * self.sr1]))
        self.init_coeff()
        self.coeff.requires_grad = False

    def init_coeff(self):
        ## use sparse coding to initialize coeff and bases
        data = self.base_layer.state_dict()['weight'] # shape: cout, cin
        data = rearrange(data, "(o p) (i q) -> (o i) (p q)", p=self.sr1, q=self.sr0)
        coeff, bases = solve_lasso(data, self.m * self.sr0 * self.sr1, alpha=1e-3)
        self.coeff.data = rearrange(coeff, "(o i) m -> o i m", o=self.out_features // self.sr1)

        # ## DEBUG
        print("||AD - W||: ", torch.dist(data, coeff @ bases).item(), 
              "; ||W||: ", torch.norm(data).item(), 
              "; ||A||: ", torch.norm(coeff).item(),
              "; ||D||: ", torch.norm(bases).item())
    
    def forward(self, x: torch.Tensor):
        base_out = self.base_layer(x)
        
        weight = torch.einsum('cvm, mki-> cvki', self.coeff.view(self.out_features // self.sr1, self.in_features // self.sr0, -1), 
                            self.bases).reshape(self.out_features // self.sr1, self.in_features // self.sr0, 
                            self.sr1, self.sr0).permute(0, 2, 1, 4).reshape(self.out_features, self.in_features)
        dcf_out = F.Linear(x, weight)
        
        return base_out + dcf_out

