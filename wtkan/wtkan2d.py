import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from .util import wavelet
from .util.KAN import KANLinear

class WTKan2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1',hidden_dim=128):
        #卷积核大小，默认为5，wt_type: 小波类型，默认为'db1'(Daubechies-1小波)
        super(WTKan2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels#输入通道数
        self.wt_levels = wt_levels#小波变换的层级数，默认为1
        self.stride = stride#步长，默认为1
        self.dilation = 1

        #小波滤波器，创建小波正变换和逆变换的滤波器，将滤波器设置为不需要梯度的参数
        self.wt_filter, self.iwt_filter = wavelet.create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        #小波变换和逆变换
        self.wt_function = partial(wavelet.wavelet_transform, filters = self.wt_filter)
        self.iwt_function = partial(wavelet.inverse_wavelet_transform, filters = self.iwt_filter)
        #基础卷积层
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels, bias=bias)
        # 将小波变换后的特征通过 KANLinear 处理
        self.hsi_kan = KANLinear(in_channels * 8, in_channels * 8)  # 用于处理小波低频和高频特征
        self.msi_kan = KANLinear(in_channels * 4, in_channels * 4)  # 用于处理高频特征
        self.msi_kan2 = KANLinear(in_channels * 2, in_channels * 2)  # 用于处理高频特征


        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride, groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):
        # 1) 输入到 Conv → KAN 分支
        print(f"[Input]                           x: {x.shape}")  
        # （图中最上方 Conv → KAN，生成 Z^(1) 的分支，这里只展示 Conv 后的形状）
        kan0 = self.base_conv(x)
        print(f"[Level 0 · BaseConv]             {kan0.shape}")  

        # 为小波分解和 KAN 分支做初始化
        curr_x2wt = x
        curr_x2kan = x
        kan_res   = []
        kan_wts   = []

        # 2) 自顶向下 Wavelet 分解 + 子带 KAN
        for i in range(self.wt_levels):
            # 2.1) Conv 分支（跳跃连接）
            kan_curr = self.base_conv(curr_x2kan)
            kan_res.append(kan_curr)
            print(f"[Level {i+1} · SkipConv]          {kan_curr.shape}")

            # 2.2) WT → 得到 4 个子带 (LL, LH, HL, HH)
            wt_out = self.wt_function(curr_x2wt)  
            # wt_out.shape = (B, C, 4, H/2, W/2)
            print(f"[Level {i+1} · WT raw]             {wt_out.shape}")

            # 只取 low/high 两路进一步处理（示例取 LL 作为 low，高频合并为一张图，实际可分开）
            x_ll = wt_out[:, :, 0, :, :]         # LL 分量
            x_h  = wt_out[:, :, 1, :, :]         # 这里简化成 one high
            print(f"    ┗ LL subband:                  {x_ll.shape}")
            print(f"    ┗ HF subband:                  {x_h.shape}")

            curr_x2wt  = wt_out
            curr_x2kan = wt_out

            # 2.3) 不同尺寸通道入 MSI_KAN
            if x_h.shape[-1] == 128:
                z = self.msi_kan(wt_out)
            else:
                z = self.msi_kan2(wt_out)
            kan_wts.append(z)
            print(f"[Level {i+1} · Subband-KAN]        {z.shape}")

        # 3) 自底向上 IWT 重建
        curr = kan_wts[-1]
        for i in range(self.wt_levels - 1, -1, -1):
            print(f"[Level {i+1} · Before IWT]         {curr.shape}")
            iwt_out = self.iwt_function(curr)  # 逆小波合并
            print(f"[Level {i+1} · IWT output]         {iwt_out.shape}")

            # 与对应的 skip-Conv 输出逐元素相加
            skip = kan_res[i]
            # 可能需要 unsqueeze：根据维度自动处理
            if iwt_out.dim() != skip.dim():
                iwt_out = iwt_out.unsqueeze(2)
            curr = iwt_out + skip
            print(f"[Level {i+1} · SkipAdd]            {curr.shape}")

        print(f"[Output]                          {curr.shape}")
        return curr




