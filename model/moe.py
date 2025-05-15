# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
import scipy
from . import thops
from .vmamba_efficient import CrossCNNBlock_Base
from .functions import LayerNorm
from .ablation_window_attn import SwinTransFormerCrossBlock
from wtconv import WTConv2d
from core.layer.kanlayer import FourierVSSBlock





class GateNetwork(nn.Module):
    def __init__(self, input_size, num_experts, top_k):
        super(GateNetwork, self).__init__()
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.input_size = input_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.fc0 = nn.Linear(input_size,num_experts)
        self.fc1 = nn.Linear(input_size, num_experts)
        self.relu1 = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)
        init.zeros_(self.fc1.weight)
        self.sp = nn.Softplus()

    def forward(self, x):
        # Flatten the input tensor
        x = self.gmp(x)+self.gap(x) # b,c 1,1,
        # print('x1.shape={}'.format(x.shape))
        x = x.view(-1, self.input_size) # 
        # print('x2.shape={}'.format(x.shape))
        inp = x
        # Pass the input through the gate network layers
        x = self.fc1(x)
        x= self.relu1(x)
        noise = self.sp(self.fc0(inp))
        noise_mean = torch.mean(noise,dim=1)
        noise_mean = noise_mean.view(-1,1)
        std = torch.std(noise,dim=1)
        std = std.view(-1,1)
        # Avoid division by zero
        std[std==0]=1
        noram_noise = (noise-noise_mean)/std
        # print('noram noise.shape={}'.format(noram_noise.shape))
        # print('x4.shape={}'.format(x.shape))
        # Apply topK operation to get the highest K values and indices along dimension 1 (columns)
        topk_values, topk_indices = torch.topk(x+noram_noise, k=self.top_k, dim=1)
        # print('topk_values={}, topk_indices={}'.format(topk_values, topk_indices))
        # Set all non-topK values to -inf to ensure they are not selected by softmax
        mask = torch.zeros_like(x).scatter_(dim=1, index=topk_indices, value=1.0)
        # x[~mask.bool()] = float('-inf')
        x[~mask.bool()] = float('-1e9')
        # Pass the masked tensor through softmax to get gating coefficients for each expert network
        gating_coeffs = self.softmax(x)
        # print('gating_coeffs={}'.format(gating_coeffs))
        return gating_coeffs






class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)

        elif mode == 'denorm':
            x = self._denormalize(x)

        else:
            raise NotImplementedError

        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean

        return x


class FuseNet3(nn.Module):
    def __init__(self, var_num, num_experts=4, drop=0.1, revin_affine=True):
        super(FuseNet3, self).__init__()
        self.var_num = var_num
        self.num_experts = num_experts
        self.drop = drop
        self.revin_affine = revin_affine

        self.gate = nn.Linear(32, self.num_experts)
        self.softmax = nn.Softmax(dim=-1)

        self.feature_extractor = FourierVSSBlock(hidden_dim=32)




        self.dropout = nn.Dropout(drop)
        self.rev = RevIN(var_num, affine=revin_affine)
        self.pre_fuse = nn.Sequential(
            InvBlock(HinResBlock, 2 * var_num, var_num),
            nn.Conv2d(2 * var_num, var_num, 1, 1, 0)
        )
        self.relu = nn.LeakyReLU()



    def forward(self, a, b):

        output = self.feature_extractor(a, b)

        return output




class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, d = 1, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1, d)
        self.G = subnet_constructor(self.split_len1, self.split_len2, d)
        self.H = subnet_constructor(self.split_len1, self.split_len2, d)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        # if not rev:
        # invert1x1conv
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=False):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        if not LU_decomposed:
            # Sample a random orthogonal matrix:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)

            self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
            self.l_mask = torch.Tensor(l_mask)
            self.eye = torch.Tensor(eye)
        self.w_shape = w_shape
        self.LU = LU_decomposed

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        if not self.LU:
            pixels = thops.pixels(input)
            dlogdet = torch.slogdet(self.weight)[1] * pixels
            if not reverse:
                weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
            else:
                weight = torch.inverse(self.weight.double()).float()\
                              .view(w_shape[0], w_shape[1], 1, 1)
            return weight, dlogdet
        else:
            self.p = self.p.to(input.device)
            self.sign_s = self.sign_s.to(input.device)
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)
            l = self.l * self.l_mask + self.eye
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
            dlogdet = thops.sum(self.log_s) * thops.pixels(input)
            if not reverse:
                w = torch.matmul(self.p, torch.matmul(l, u))
            else:
                l = torch.inverse(l.double()).float()
                u = torch.inverse(u.double()).float()
                w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
            return w.view(w_shape[0], w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)
        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet


class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.conv_1 = WTConv2d(
            in_channels=in_size,
            out_channels=out_size,
            wt_levels=2,
            kernel_size=3,
            stride=1,
            bias=True
        )
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = WTConv2d(
            in_channels=in_size,
            out_channels=out_size,
            wt_levels=2,
            kernel_size=3,
            stride=1,
            bias=True
        )
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        return x+resi


    def __init__(self,basefilter) -> None:
    # def __init__(self,basefilter,is_2d=False) -> None:
        super().__init__()
        self.nc = basefilter
        # self.is_2d = is_2d
    def forward(self, x, x_size):
        B,HW,C = x.shape
        x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])  # B Ph*Pw C
        return x
        # if self.is_2d:
        #     f0 = x[0].transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])
        #     f1 = x[1].transpose(1, 2).view(B, self.nc, x_size[0], x_size[1]).transpose(2, 3).flip(2)
        #     f2 = x[2].flip(1).transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])
        #     f3 = x[3].flip(1).transpose(1, 2).view(B, self.nc, x_size[0], x_size[1]).transpose(2, 3).flip(2)
        #     return (f0+f1+f2+f3)/4
        # else:
        #     x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])  # B Ph*Pw C
        #     return x


    """ 2D Image to Patch Embedding
    """
    # def __init__(self,patch_size=4, stride=4,in_chans=36, embed_dim=32*32*32, norm_layer=None, flatten=True, is_2d=False):
    def __init__(self,patch_size=4, stride=4,in_chans=36, embed_dim=32*32*32, norm_layer=None, flatten=True):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = LayerNorm(embed_dim,'BiasFree')
        # self.is_2d = is_2d

    def forward(self, x):
        #（b,c,h,w)->(b,c*s*p,h//s,w//s)
        #(b,h*w//s**2,c*s**2)
        B, C, H, W = x.shape
        # x = F.unfold(x, self.patch_size, stride=self.patch_size)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            # x = self.norm(x)
        return x
        # if self.is_2d:
        #     if self.flatten:
        #         x_reverse = x.transpose(2, 3).flip(2)
        #         x0 = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        #         x1 = x_reverse.flatten(2).transpose(1, 2)
        #         x2 = x0.flip(1)
        #         x3 = x1.flip(1)
        #     return (x0, x1, x2, x3)
        # else:
        #     if self.flatten:
        #         x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        #     # x = self.norm(x)
        #     return x



    def __init__(self, channels, num_experts=4, k=2):
        super(FuseNet_CNN, self).__init__()
        self.gate = GateNetwork(channels, num_experts, k)
        self.num_experts = num_experts
        self.expert_networks_d = nn.ModuleList(
            [CrossCNNBlock_Base(hidden_dim=channels) for i in range(num_experts)])
        self.pre_fuse = nn.Sequential(InvBlock(HinResBlock, 2*channels, channels),
                                         nn.Conv2d(2*channels,channels,1,1,0))
        self.relu = nn.LeakyReLU()

    def forward(self, a, b, infer=False):
        # a=a.permute(0,3,1,2)
        # b=b.permute(0,3,1,2)
        x = self.pre_fuse(torch.cat((a, b), dim=1))
        x = self.relu(x)
        # print('x.shape={}'.format(x.shape))
        cof = self.gate(x)
        # print('cof={}'.format(cof))
        # print('cof.shape={}'.format(cof.shape))
        out = torch.zeros_like(x).to(x.device)
        if infer:
            
            for idx in range(self.num_experts):
                if cof[:,idx].all()==0:
                    continue
                # print('mask_all={}'.format(torch.where(cof[:,idx]>0)))
                mask = torch.where(cof[:,idx]>0)[0]
                # print('mask={}'.format(mask))
                expert_layer = self.expert_networks_d[idx]
                # print('a[mask].shape={}'.format(a[mask].shape))
                expert_out = expert_layer(a[mask], b[mask]).permute(0,3,1,2)
                cof_k = cof[mask,idx].view(-1,1,1,1)
                # print(cof_k)
                # print('out[mask].shape={}, expert_out.shape={}, cof_k.shape={}'.format(out[mask].shape, expert_out.shape, cof_k.shape))
                out[mask]+=expert_out*cof_k
            # print('cof_k={}'.format(cof_k))
            return out
        else:
            for idx in range(self.num_experts):
                expert_layer = self.expert_networks_d[idx]
                expert_out = expert_layer(a, b).permute(0,3,1,2)
                weighted_expert_outputs = cof[:,idx] * expert_out
                # print(cof_k)
                # print('out[mask].shape={}, expert_out.shape={}, cof_k.shape={}'.format(out[mask].shape, expert_out.shape, cof_k.shape))
                out += weighted_expert_outputs
            # print('cof_k={}'.format(cof_k))
            return out, cof



    def __init__(self, channels, num_experts=4, k=2):
        super(FuseNet_Swin, self).__init__()
        self.gate = GateNetwork(channels, num_experts, k)
        self.num_experts = num_experts
        self.expert_networks_d = nn.ModuleList(
            [SwinTransFormerCrossBlock(channels,4) for i in range(num_experts)])
        self.pre_fuse = nn.Sequential(InvBlock(HinResBlock, 2*channels, channels),
                                         nn.Conv2d(2*channels,channels,1,1,0))
        self.relu = nn.LeakyReLU()

    def forward(self, a, b, infer=False):
        # a=a.permute(0,3,1,2)
        # b=b.permute(0,3,1,2)
        x = self.pre_fuse(torch.cat((a, b), dim=1))
        x = self.relu(x)
        # print('x.shape={}'.format(x.shape))
        cof = self.gate(x)
        # print('cof={}'.format(cof))
        # print('cof.shape={}'.format(cof.shape))
        out = torch.zeros_like(x).to(x.device)
        if infer:
            
            for idx in range(self.num_experts):
                if cof[:,idx].all()==0:
                    continue
                # print('mask_all={}'.format(torch.where(cof[:,idx]>0)))
                mask = torch.where(cof[:,idx]>0)[0]
                # print('mask={}'.format(mask))
                expert_layer = self.expert_networks_d[idx]
                # print('a[mask].shape={}'.format(a[mask].shape))
                expert_out = expert_layer(a[mask], b[mask]).permute(0,3,1,2)
                cof_k = cof[mask,idx].view(-1,1,1,1)
                # print(cof_k)
                # print('out[mask].shape={}, expert_out.shape={}, cof_k.shape={}'.format(out[mask].shape, expert_out.shape, cof_k.shape))
                out[mask]+=expert_out*cof_k
            # print('cof_k={}'.format(cof_k))
            return out
        else:
            for idx in range(self.num_experts):
                expert_layer = self.expert_networks_d[idx]
                expert_out = expert_layer(a, b).permute(0,3,1,2)
                weighted_expert_outputs = cof[:,idx] * expert_out
                # print(cof_k)
                # print('out[mask].shape={}, expert_out.shape={}, cof_k.shape={}'.format(out[mask].shape, expert_out.shape, cof_k.shape))
                out += weighted_expert_outputs
            # print('cof_k={}'.format(cof_k))
            return out, cof



