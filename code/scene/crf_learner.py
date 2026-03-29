from typing import Union
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
# import matplotlib as plt
import os
# import scio
img2mse = lambda x, y, z : torch.mean((x - y) ** 2 * z)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
tonemap = lambda x : (np.log(np.clip(x,0,1) * 5000 + 1 ) / np.log(5000 + 1) * 255).astype(np.uint8)

class CRFLearner(nn.Module):
    def __init__(
            self,
            fix_expourse_time=None,
            n_input_dims: int = 1,
            n_grayscale_factors: int = 3,
            n_gammas: int = 1,
            n_neurons: int = 32,
            n_hidden_layers: int = 2,
            n_frequencies: int = 4,
            grayscale_factors_activation: str = "Sigmoid",
            gamma_activation: str = "Softplus",
    ) -> None:
        super().__init__()
        W = 256
        self.exps_linears_r = nn.ModuleList([nn.Linear(1, W//2)])
        self.exps_linears_g = nn.ModuleList([nn.Linear(1, W//2)])
        self.exps_linears_b = nn.ModuleList([nn.Linear(1, W//2)])
        self.b_l_linner = nn.Linear(W//2, 1)
        self.r_l_linner = nn.Linear(W//2, 1)
        self.g_l_linner = nn.Linear(W//2, 1)
        self.split_rgb = True
        self.fixCRF = True
        self.fixExps = False
        if fix_expourse_time is not None:
            self.fix_expourse_time = torch.tensor([fix_expourse_time]).cuda()
        else:
            self.fix_expourse_time = None
        self.gamma_r = nn.Parameter(torch.tensor(1/2.2))
        self.gamma_g = nn.Parameter(torch.tensor(1/2.2))
        self.gamma_b = nn.Parameter(torch.tensor(1/2.2))
        self.exposure = nn.Parameter(torch.tensor(1.0))
    def forward(self, x, expourse_time):
        if self.fixCRF:
            if not self.split_rgb:
                if self.fixExps:
                    x = x + torch.log(expourse_time)
                    ldr = (torch.exp(x) / (torch.exp(x) + 1) ) ** (self.gamma_r)
                else:
                    x = x + torch.log(self.exposure)
                    ldr = (torch.exp(x) / (torch.exp(x) + 1) ) ** (self.gamma_r)
            else:
                if self.fixExps:
                    r_h = x[:,0:1] + torch.log(expourse_time)
                    g_h = x[:,1:2] + torch.log(expourse_time)
                    b_h = x[:,2:3] + torch.log(expourse_time)
                    r_l = (torch.exp(r_h) / (torch.exp(r_h) + 1) ) ** (self.gamma_r)
                    g_l = (torch.exp(g_h) / (torch.exp(g_h) + 1) ) ** (self.gamma_g)
                    b_l = (torch.exp(b_h) / (torch.exp(b_h) + 1) ) ** (self.gamma_b)
                    ldr = torch.cat([r_l, g_l, b_l], -1)
                else:
                    r_h = x[:,0:1] + torch.log(self.exposure)
                    g_h = x[:,1:2] + torch.log(self.exposure)
                    b_h = x[:,2:3] + torch.log(self.exposure)
                    r_l = (torch.exp(r_h) / (torch.exp(r_h) + 1) ) ** (self.gamma_r)
                    g_l = (torch.exp(g_h) / (torch.exp(g_h) + 1) ) ** (self.gamma_g)
                    b_l = (torch.exp(b_h) / (torch.exp(b_h) + 1) ) ** (self.gamma_b)
                    ldr = torch.cat([r_l, g_l, b_l], -1)
        else:
            # print(expourse_time)
        # x = torch.log(x + 1e-8)
            if self.fix_expourse_time is not None:
                # print('fix_expourse_time')
                r_h = x[:,0:1] + torch.log(self.fix_expourse_time)
                g_h = x[:,1:2] + torch.log(self.fix_expourse_time)
                b_h = x[:,2:3] + torch.log(self.fix_expourse_time)
            elif expourse_time is not None:
                if self.fixExps:
                    r_h = x[:,0:1] + torch.log(expourse_time)
                    g_h = x[:,1:2] + torch.log(expourse_time)
                    b_h = x[:,2:3] + torch.log(expourse_time)
                else:
                    r_h = x[:,0:1] + torch.log(self.exposure)
                    g_h = x[:,1:2] + torch.log(self.exposure)
                    b_h = x[:,2:3] + torch.log(self.exposure)
            else:
                r_h = x[:,0:1] 
                g_h = x[:,1:2] 
                b_h = x[:,2:3] 
            # print('r_h', r_h.shape)
            if not self.split_rgb:
                for i, l in enumerate(self.exps_linears_r):
                    r_h = self.exps_linears_r[i](r_h)
                    r_h = F.relu(r_h)
                r_l = self.r_l_linner(r_h)
                
                for i, l in enumerate(self.exps_linears_r):
                    g_h = self.exps_linears_r[i](g_h)
                    g_h = F.relu(g_h)
                g_l = self.r_l_linner(g_h)
      
                for i, l in enumerate(self.exps_linears_r):
                    b_h = self.exps_linears_r[i](b_h)
                    b_h = F.relu(b_h)
                b_l = self.r_l_linner(b_h)
 
            else:
                for i, l in enumerate(self.exps_linears_r):
                    r_h = self.exps_linears_r[i](r_h)
                    r_h = F.relu(r_h)
                r_l = self.r_l_linner(r_h)
  
                for i, l in enumerate(self.exps_linears_g):
                    g_h = self.exps_linears_g[i](g_h)
                    g_h = F.relu(g_h)
                g_l = self.g_l_linner(g_h)
    
                for i, l in enumerate(self.exps_linears_b):
                    b_h = self.exps_linears_b[i](b_h)
                    b_h = F.relu(b_h)
                b_l = self.b_l_linner(b_h)
            
            ldr = torch.cat([r_l, g_l, b_l], -1)
            ldr = torch.sigmoid(ldr)
            
        return ldr.to(torch.float)
    
    def point_constraint_fixCRF(self, input_value, gt):
        ln_x = torch.ones([3,1]) * input_value
        ln_x = ln_x.cuda()
        if not self.split_rgb:
            rgb_l = (torch.exp(ln_x) / (torch.exp(ln_x) + 1) ) ** (self.gamma_r)
        else:
            r_h = ln_x
            g_h = ln_x
            b_h = ln_x
            r_l = (torch.exp(r_h) / (torch.exp(r_h) + 1) ) ** (self.gamma_r)
            g_l = (torch.exp(g_h) / (torch.exp(g_h) + 1) ) ** (self.gamma_g)
            b_l = (torch.exp(b_h) / (torch.exp(b_h) + 1) ) ** (self.gamma_b)
            rgb_l = torch.cat([r_l, g_l, b_l], -1)
        
        return img2mse(rgb_l, gt, 1)
    
    def point_constraint(self, input_value, gt):
    
        ln_x = torch.ones([3,1]) * input_value
        ln_x = ln_x.cuda()

        r_h = ln_x
        g_h = ln_x
        b_h = ln_x
        if not self.split_rgb:
            for i, l in enumerate(self.exps_linears_r):
                r_h = self.exps_linears_r[i](r_h)
                r_h = F.relu(r_h)
            r_l = self.r_l_linner(r_h)
            # r_l = torch.sigmoid(r_l)
            for i, l in enumerate(self.exps_linears_r):
                g_h = self.exps_linears_r[i](g_h)
                g_h = F.relu(g_h)
            g_l = self.r_l_linner(g_h)
            # g_l = torch.sigmoid(g_l)
            for i, l in enumerate(self.exps_linears_r):
                b_h = self.exps_linears_r[i](b_h)
                b_h = F.relu(b_h)
            b_l = self.r_l_linner(b_h)
        else:    
            for i, l in enumerate(self.exps_linears_r):
                r_h = self.exps_linears_r[i](r_h)
                r_h = F.relu(r_h)
            r_l = self.r_l_linner(r_h)
            # r_l = torch.sigmoid(r_l)
            for i, l in enumerate(self.exps_linears_g):
                g_h = self.exps_linears_g[i](g_h)
                g_h = F.relu(g_h)
            g_l = self.g_l_linner(g_h)
            # g_l = torch.sigmoid(g_l)
            for i, l in enumerate(self.exps_linears_b):
                b_h = self.exps_linears_b[i](b_h)
                b_h = F.relu(b_h)
            b_l = self.b_l_linner(b_h)
            # b_l = torch.sigmoid(b_l)
        rgb_l = torch.cat([r_l, g_l, b_l], -1)
        rgb_l = torch.sigmoid(rgb_l)
        return img2mse(rgb_l, gt, 1)

    def white_balance_constraint(self, begin, end):
        ln_x = torch.linspace(begin, end, 1000).reshape([-1, 1]).cuda()
        r_h = ln_x
        g_h = ln_x
        b_h = ln_x
        for i, l in enumerate(self.exps_linears_r):
            r_h = self.exps_linears_r[i](r_h)
            r_h = F.relu(r_h)
        r_l = self.r_l_linner(r_h)
        r_l = torch.sigmoid(r_l)
        r_l = r_l.mean()
        for i, l in enumerate(self.exps_linears_g):
            g_h = self.exps_linears_g[i](g_h)
            g_h = F.relu(g_h)
        g_l = self.g_l_linner(g_h)
        g_l = torch.sigmoid(g_l)
        g_l = g_l.mean()
        for i, l in enumerate(self.exps_linears_b):
            b_h = self.exps_linears_b[i](b_h)
            b_h = F.relu(b_h)
        b_l = self.b_l_linner(b_h)
        b_l = torch.sigmoid(b_l)
        b_l = b_l.mean()
        # print('b_l', b_l.shape)
        return (img2mse(r_l, g_l, 1) + img2mse(r_l, b_l, 1) + img2mse(g_l, b_l, 1)) / 3
    
    
    
    def get_CRF_grad(self):
        with torch.enable_grad():
            ln_x = torch.linspace(-5, 5, 1000).reshape([-1, 1]).cuda()
            ln_x.requires_grad_(True)
            r_h = ln_x
            g_h = ln_x
            b_h = ln_x
            if not self.split_rgb:
                for i, l in enumerate(self.exps_linears_r):
                    r_h = self.exps_linears_r[i](r_h)
                    r_h = F.relu(r_h)
                r_l = self.r_l_linner(r_h)
                # r_l = torch.sigmoid(r_l)
                for i, l in enumerate(self.exps_linears_r):
                    g_h = self.exps_linears_r[i](g_h)
                    g_h = F.relu(g_h)
                g_l = self.r_l_linner(g_h)
                # g_l = torch.sigmoid(g_l)
                for i, l in enumerate(self.exps_linears_r):
                    b_h = self.exps_linears_r[i](b_h)
                    b_h = F.relu(b_h)
                b_l = self.r_l_linner(b_h)
            else:
                for i, l in enumerate(self.exps_linears_r):
                    r_h = self.exps_linears_r[i](r_h)
                    r_h = F.relu(r_h)
                r_l = self.r_l_linner(r_h)
                
                for i, l in enumerate(self.exps_linears_g):
                    g_h = self.exps_linears_g[i](g_h)
                    g_h = F.relu(g_h)
                g_l = self.g_l_linner(g_h)

                for i, l in enumerate(self.exps_linears_b):
                    b_h = self.exps_linears_b[i](b_h)
                    b_h = F.relu(b_h)
                b_l = self.b_l_linner(b_h)
                
            # b_l = torch.sigmoid(b_l)
            # rgb_l = torch.cat([r_l, g_l, b_l], -1)
            # rgb_l = torch.sigmoid(rgb_l)
            crf_output = torch.sigmoid(torch.cat([r_l, g_l, b_l], -1))
            
            # crf_output = torch.sigmoid(torch.cat([r_l, r_l, r_l], -1))
            crf_grad = torch.autograd.grad(
                outputs=crf_output,
                inputs=ln_x,
                grad_outputs=torch.ones_like(crf_output, requires_grad=False),
                retain_graph=True,
                create_graph=True)[0]    
            res = F.relu(-crf_grad)
        return res.mean()
