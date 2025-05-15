import math
import torch.fft as fft
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Any
from functools import partial
import torch.utils.checkpoint as checkpoint
from wtkan.util.KAN import KANLinear
from einops import rearrange

def B_batch(x, grid, k=0, extend=True, device='cpu'):
    '''
    evaludate x on B-spline bases

    Args:
    -----
        x : 2D torch.tensor
            inputs, shape (number of splines, number of samples)
        grid : 2D torch.tensor
            grids, shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
        extend : bool
            If True, k points are extended on both ends. If False, no extension (zero boundary condition). Default: True
        device : str
            devicde

    Returns:
    --------
        spline values : 3D torch.tensor
            shape (number of splines, number of B-spline bases (coeffcients), number of samples). The numbef of B-spline bases = number of grid points + k - 1.

    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x = torch.normal(0,1,size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    >>> B_batch(x, grids, k=k).shape
    torch.Size([5, 13, 100])
    '''

    # x shape: (size, x); grid shape: (size, grid)
    def extend_grid(grid, k_extend=0):
        # pad k to left and right
        # grid shape: (batch, grid)
        h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

        for i in range(k_extend):
            grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
            grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
        grid = grid.to(device)
        return grid

    if extend == True:
        grid = extend_grid(grid, k_extend=k)

    grid = grid.unsqueeze(dim=2).to(device)
    x = x.unsqueeze(dim=1).to(device)

    if k == 0:
        value = (x >= grid[:, :-1]) * (x < grid[:, 1:])
    else:
        B_km1 = B_batch(x[:, 0], grid=grid[:, :, 0], k=k - 1, extend=False, device=device)
        value = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * B_km1[:, :-1] + (
                grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * B_km1[:, 1:]
    return value


def coef2curve(x_eval, grid, coef, k, device="cpu"):
    '''
    converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).

    Args:
    -----
        x_eval : 2D torch.tensor)
            shape (number of splines, number of samples)
        grid : 2D torch.tensor)
            shape (number of splines, number of grid points)
        coef : 2D torch.tensor)
            shape (number of splines, number of coef params). number of coef params = number of grid intervals + k
        k : int
            the piecewise polynomial order of splines.
        device : str
            devicde

    Returns:
    --------
        y_eval : 2D torch.tensor
            shape (number of splines, number of samples)

    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x_eval = torch.normal(0,1,size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    >>> coef = torch.normal(0,1,size=(num_spline, num_grid_interval+k))
    >>> coef2curve(x_eval, grids, coef, k=k).shape
    torch.Size([5, 100])
    '''
    # x_eval: (size, batch), grid: (size, grid), coef: (size, coef)
    # coef: (size, coef), B_batch: (size, coef, batch), summer over coef
    if coef.dtype != x_eval.dtype:
        coef = coef.to(x_eval.dtype)
    y_eval = torch.einsum('ij,ijk->ik', coef, B_batch(x_eval, grid, k, device=device))
    return y_eval


def curve2coef(x_eval, y_eval, grid, k, device="cpu"):
    '''
    converting B-spline curves to B-spline coefficients using least squares.

    Args:
    -----
        x_eval : 2D torch.tensor
            shape (number of splines, number of samples)
        y_eval : 2D torch.tensor
            shape (number of splines, number of samples)
        grid : 2D torch.tensor
            shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
        device : str
            devicde

    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x_eval = torch.normal(0,1,size=(num_spline, num_sample))
    >>> y_eval = torch.normal(0,1,size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    torch.Size([5, 13])
    '''
    # x_eval: (size, batch); y_eval: (size, batch); grid: (size, grid); k: scalar
    mat = B_batch(x_eval, grid, k, device=device).permute(0, 2, 1)
    # coef = torch.linalg.lstsq(mat, y_eval.unsqueeze(dim=2)).solution[:, :, 0]
    coef = torch.linalg.lstsq(mat.to(device), y_eval.unsqueeze(dim=2).to(device),
                              driver='gelsy' if device == 'cpu' else 'gels').solution[:, :, 0]
    return coef.to(device)


class KANLayer(nn.Module):
    """
    KANLayer class


    Attributes:
    -----------
        in_dim: int
            input dimension
        out_dim: int
            output dimension
        size: int
            the number of splines = input dimension * output dimension
        k: int
            the piecewise polynomial order of splines
        grid: 2D torch.float
            grid points
        noises: 2D torch.float
            injected noises to splines at initialization (to break degeneracy)
        coef: 2D torch.tensor
            coefficients of B-spline bases
        scale_base: 1D torch.float
            magnitude of the residual function b(x)
        scale_sp: 1D torch.float
            mangitude of the spline function spline(x)
        base_fun: fun
            residual function b(x)
        mask: 1D torch.float
            mask of spline functions. setting some element of the mask to zero means setting the corresponding activation to zero function.
        grid_eps: float in [0,1]
            a hyperparameter used in update_grid_from_samples. When grid_eps = 0, the grid is uniform; when grid_eps = 1, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
        weight_sharing: 1D tensor int
            allow spline activations to share parameters
        lock_counter: int
            counter how many activation functions are locked (weight sharing)
        lock_id: 1D torch.int
            the id of activation functions that are locked
        device: str
            device

    Methods:
    --------
        __init__():
            initialize a KANLayer
        forward():
            forward
        update_grid_from_samples():
            update grids based on samples' incoming activations
        initialize_grid_from_parent():
            initialize grids from another model
        get_subset():
            get subset of the KANLayer (used for pruning)
        lock():
            lock several activation functions to share parameters
        unlock():
            unlock already locked activation functions
    """

    def __init__(self, in_dim=3, out_dim=2, num=5, k=3, noise_scale=0.1, scale_base=1.0, scale_sp=1.0,
                 base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True,
                 device='cuda'):
        ''''
        initialize a KANLayer

        Args:
        -----
            in_dim : int
                input dimension. Default: 2.
            out_dim : int
                output dimension. Default: 3.
            num : int
                the number of grid intervals = G. Default: 5.
            k : int
                the order of piecewise polynomial. Default: 3.
            noise_scale : float
                the scale of noise injected at initialization. Default: 0.1.
            scale_base : float
                the scale of the residual function b(x). Default: 1.0.
            scale_sp : float
                the scale of the base function spline(x). Default: 1.0.
            base_fun : function
                residual function b(x). Default: torch.nn.SiLU()
            grid_eps : float
                When grid_eps = 0, the grid is uniform; when grid_eps = 1, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes. Default: 0.02.
            grid_range : list/np.array of shape (2,)
                setting the range of grids. Default: [-1,1].
            sp_trainable : bool
                If true, scale_sp is trainable. Default: True.
            sb_trainable : bool
                If true, scale_base is trainable. Default: True.
            device : str
                device

        Returns:
        --------
            self

        Example
        -------
        >>> model = KANLayer(in_dim=3, out_dim=5)
        >>> (model.in_dim, model.out_dim)
        (3, 5)
        '''
        super(KANLayer, self).__init__()
        # size
        self.size = size = out_dim * in_dim
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k

        # shape: (size, num)
        self.grid = torch.einsum('i,j->ij', torch.ones(size, device=device),
                                 torch.linspace(grid_range[0], grid_range[1], steps=num + 1, device=device))
        self.grid = torch.nn.Parameter(self.grid).requires_grad_(False)
        noises = (torch.rand(size, self.grid.shape[1]) - 1 / 2) * noise_scale / num
        noises = noises.to(device)
        # shape: (size, coef)
        self.coef = torch.nn.Parameter(curve2coef(self.grid, noises, self.grid, k, device))
        if isinstance(scale_base, float):
            self.scale_base = torch.nn.Parameter(torch.ones(size, device=device) * scale_base).requires_grad_(
                sb_trainable)  # make scale trainable
        else:
            self.scale_base = torch.nn.Parameter(torch.FloatTensor(scale_base).to(device)).requires_grad_(sb_trainable)
        self.scale_sp = torch.nn.Parameter(torch.ones(size, device=device) * scale_sp).requires_grad_(
            sp_trainable)  # make scale trainable
        self.base_fun = base_fun

        self.mask = torch.nn.Parameter(torch.ones(size, device=device)).requires_grad_(False)
        self.grid_eps = grid_eps
        self.weight_sharing = torch.arange(size)
        self.lock_counter = 0
        self.lock_id = torch.zeros(size)
        self.device = device

    def forward(self, x):
        '''
        KANLayer forward given input x

        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)

        Returns:
        --------
            y : 2D torch.float
                outputs, shape (number of samples, output dimension)
            preacts : 3D torch.float
                fan out x into activations, shape (number of sampels, output dimension, input dimension)
            postacts : 3D torch.float
                the outputs of activation functions with preacts as inputs
            postspline : 3D torch.float
                the outputs of spline functions with preacts as inputs

        Example
        -------
        >>> model = KANLayer(in_dim=3, out_dim=5)
        >>> x = torch.normal(0,1,size=(100,3))
        >>> y, preacts, postacts, postspline = model(x)
        >>> y.shape, preacts.shape, postacts.shape, postspline.shape
        (torch.Size([100, 5]),
         torch.Size([100, 5, 3]),
         torch.Size([100, 5, 3]),
         torch.Size([100, 5, 3]))
        '''
        batch = x.shape[0]
        # x: shape (batch, in_dim) => shape (size, batch) (size = out_dim * in_dim)
        x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, device=self.device)).reshape(batch,
                                                                                               self.size).permute(1, 0)
        preacts = x.permute(1, 0).clone().reshape(batch, self.out_dim, self.in_dim)
        base = self.base_fun(x).permute(1, 0)  # shape (batch, size)
        y = coef2curve(x_eval=x, grid=self.grid[self.weight_sharing], coef=self.coef[self.weight_sharing], k=self.k,
                       device=self.device)  # shape (size, batch)
        y = y.permute(1, 0)  # shape (batch, size)
        postspline = y.clone().reshape(batch, self.out_dim, self.in_dim)
        y = self.scale_base.unsqueeze(dim=0) * base + self.scale_sp.unsqueeze(dim=0) * y
        y = self.mask[None, :] * y
        postacts = y.clone().reshape(batch, self.out_dim, self.in_dim)
        y = torch.sum(y.reshape(batch, self.out_dim, self.in_dim), dim=2)  # shape (batch, out_dim)
        # y shape: (batch, out_dim); preacts shape: (batch, in_dim, out_dim)
        # postspline shape: (batch, in_dim, out_dim); postacts: (batch, in_dim, out_dim)
        # postspline is for extension; postacts is for visualization
        return y  # , preacts, postacts, postspline

    def update_grid_from_samples(self, x):
        '''
        update grid from samples

        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)

        Returns:
        --------
            None

        Example
        -------
        >>> model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(model.grid.data)
        >>> x = torch.linspace(-3,3,steps=100)[:,None]
        >>> model.update_grid_from_samples(x)
        >>> print(model.grid.data)
        tensor([[-1.0000, -0.6000, -0.2000,  0.2000,  0.6000,  1.0000]])
        tensor([[-3.0002, -1.7882, -0.5763,  0.6357,  1.8476,  3.0002]])
        '''
        batch = x.shape[0]
        x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(batch, self.size).permute(
            1, 0)
        x_pos = torch.sort(x, dim=1)[0]
        y_eval = coef2curve(x_pos, self.grid, self.coef, self.k, device=self.device)
        num_interval = self.grid.shape[1] - 1
        ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
        grid_adaptive = x_pos[:, ids]
        margin = 0.01
        grid_uniform = torch.cat(
            [grid_adaptive[:, [0]] - margin + (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]] + 2 * margin) * a for a in
             np.linspace(0, 1, num=self.grid.shape[1])], dim=1)
        self.grid.data = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k, device=self.device)

    def initialize_grid_from_parent(self, parent, x):
        '''
        update grid from a parent KANLayer & samples

        Args:
        -----
            parent : KANLayer
                a parent KANLayer (whose grid is usually coarser than the current model)
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)

        Returns:
        --------
            None

        Example
        -------
        >>> batch = 100
        >>> parent_model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(parent_model.grid.data)
        >>> model = KANLayer(in_dim=1, out_dim=1, num=10, k=3)
        >>> x = torch.normal(0,1,size=(batch, 1))
        >>> model.initialize_grid_from_parent(parent_model, x)
        >>> print(model.grid.data)
        tensor([[-1.0000, -0.6000, -0.2000,  0.2000,  0.6000,  1.0000]])
        tensor([[-1.0000, -0.8000, -0.6000, -0.4000, -0.2000,  0.0000,  0.2000,  0.4000,
          0.6000,  0.8000,  1.0000]])
        '''
        batch = x.shape[0]
        # preacts: shape (batch, in_dim) => shape (size, batch) (size = out_dim * in_dim)
        x_eval = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(batch,
                                                                                                  self.size).permute(1,
                                                                                                                     0)
        x_pos = parent.grid
        sp2 = KANLayer(in_dim=1, out_dim=self.size, k=1, num=x_pos.shape[1] - 1, scale_base=0., device=self.device)
        sp2.coef.data = curve2coef(sp2.grid, x_pos, sp2.grid, k=1, device=self.device)
        y_eval = coef2curve(x_eval, parent.grid, parent.coef, parent.k, device=self.device)
        percentile = torch.linspace(-1, 1, self.num + 1).to(self.device)
        self.grid.data = sp2(percentile.unsqueeze(dim=1))[0].permute(1, 0)
        self.coef.data = curve2coef(x_eval, y_eval, self.grid, self.k, self.device)

    def get_subset(self, in_id, out_id):
        '''
        get a smaller KANLayer from a larger KANLayer (used for pruning)

        Args:
        -----
            in_id : list
                id of selected input neurons
            out_id : list
                id of selected output neurons

        Returns:
        --------
            spb : KANLayer

        Example
        -------
        >>> kanlayer_large = KANLayer(in_dim=10, out_dim=10, num=5, k=3)
        >>> kanlayer_small = kanlayer_large.get_subset([0,9],[1,2,3])
        >>> kanlayer_small.in_dim, kanlayer_small.out_dim
        (2, 3)
        '''
        spb = KANLayer(len(in_id), len(out_id), self.num, self.k, base_fun=self.base_fun, device=self.device)
        spb.grid.data = self.grid.reshape(self.out_dim, self.in_dim, spb.num + 1)[out_id][:, in_id].reshape(-1,
                                                                                                            spb.num + 1)
        spb.coef.data = self.coef.reshape(self.out_dim, self.in_dim, spb.coef.shape[1])[out_id][:, in_id].reshape(-1,
                                                                                                                  spb.coef.shape[
                                                                                                                      1])
        spb.scale_base.data = self.scale_base.reshape(self.out_dim, self.in_dim)[out_id][:, in_id].reshape(-1, )
        spb.scale_sp.data = self.scale_sp.reshape(self.out_dim, self.in_dim)[out_id][:, in_id].reshape(-1, )
        spb.mask.data = self.mask.reshape(self.out_dim, self.in_dim)[out_id][:, in_id].reshape(-1, )

        spb.in_dim = len(in_id)
        spb.out_dim = len(out_id)
        spb.size = spb.in_dim * spb.out_dim
        return spb

    def lock(self, ids):
        '''
        lock activation functions to share parameters based on ids

        Args:
        -----
            ids : list
                list of ids of activation functions

        Returns:
        --------
            None

        Example
        -------
        >>> model = KANLayer(in_dim=3, out_dim=3, num=5, k=3)
        >>> print(model.weight_sharing.reshape(3,3))
        >>> model.lock([[0,0],[1,2],[2,1]]) # set (0,0),(1,2),(2,1) functions to be the same
        >>> print(model.weight_sharing.reshape(3,3))
        tensor([[0, 1, 2],
                [3, 4, 5],
                [6, 7, 8]])
        tensor([[0, 1, 2],
                [3, 4, 0],
                [6, 0, 8]])
        '''
        self.lock_counter += 1
        # ids: [[i1,j1],[i2,j2],[i3,j3],...]
        for i in range(len(ids)):
            if i != 0:
                self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]] = ids[0][1] * self.in_dim + ids[0][0]
            self.lock_id[ids[i][1] * self.in_dim + ids[i][0]] = self.lock_counter

    def unlock(self, ids):
        '''
        unlock activation functions

        Args:
        -----
            ids : list
                list of ids of activation functions

        Returns:
        --------
            None

        Example
        -------
        >>> model = KANLayer(in_dim=3, out_dim=3, num=5, k=3)
        >>> model.lock([[0,0],[1,2],[2,1]]) # set (0,0),(1,2),(2,1) functions to be the same
        >>> print(model.weight_sharing.reshape(3,3))
        >>> model.unlock([[0,0],[1,2],[2,1]]) # unlock the locked functions
        >>> print(model.weight_sharing.reshape(3,3))
        tensor([[0, 1, 2],
                [3, 4, 0],
                [6, 0, 8]])
        tensor([[0, 1, 2],
                [3, 4, 5],
                [6, 7, 8]])
        '''
        # check ids are locked
        num = len(ids)
        locked = True
        for i in range(num):
            locked *= (self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]] == self.weight_sharing[
                ids[0][1] * self.in_dim + ids[0][0]])
        if locked == False:
            print("they are not locked. unlock failed.")
            return 0
        for i in range(len(ids)):
            self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]] = ids[i][1] * self.in_dim + ids[i][0]
            self.lock_id[ids[i][1] * self.in_dim + ids[i][0]] = 0
        self.lock_counter -= 1




    def __init__(
        self,
        hidden_dim: int = 32,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        gridsize: int = 10,  # Fourier 变换的网格大小
        addbias: bool = True,  # 是否添加偏置
        use_checkpoint: bool = False,  # 是否使用梯度检查点
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # 定义 Fourier KAN 融合层
        self.fourier_layer_x = NaiveFourierKANLayer(
            inputdim=hidden_dim, outdim=hidden_dim, gridsize=gridsize, addbias=addbias
        )
        self.fourier_layer_b = NaiveFourierKANLayer(
            inputdim=hidden_dim, outdim=hidden_dim, gridsize=gridsize, addbias=addbias
        )

        # 归一化层
        self.norm = norm_layer(hidden_dim)

    def _forward(self, input_x: torch.Tensor, input_b: torch.Tensor):
        # 输入形状调整：从 [B, C, H, W] -> [B, H, W, C]
        x, b = input_x.permute(0, 2, 3, 1), input_b.permute(0, 2, 3, 1)

        # 分别通过 Fourier KAN 层处理
        fused_x = self.fourier_layer_x(x)
        fused_b = self.fourier_layer_b(b)
        #print("fused_b",fused_b.shape)
        # 融合特征（加和方式，也可尝试拼接）
        fused = fused_x + fused_b

        # 归一化处理
        fused = self.norm(fused)

        # 输出形状调整：从 [B, H, W, C] -> [B, C, H, W]
        return fused.permute(0, 3, 1, 2)

    def forward(self, input_x: torch.Tensor, input_b: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint(self._forward, input_x, input_b)
        else:
            return self._forward(input_x, input_b)

class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_3 = nn.Conv2d(in_size + in_size, out_size, 3, 1, 1)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
            self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        return x + resi





class FourierVSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 32,
        drop_path: float = 0,
        norm_layer: callable = partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint: bool = False,
        gridsize=10,
        addbias=True,
        num_blocks=3,
        
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # 定义处理模块
        self.norm = norm_layer(hidden_dim)
        # self.fourier_layer_x1 = NaiveFourierKANLayer(
        #     inputdim=hidden_dim, outdim=hidden_dim, gridsize=gridsize, addbias=addbias
        # )
        self.kan = KANLinear(hidden_dim * 2, hidden_dim * 2)
        self.lrelu = nn.LeakyReLU(inplace=True)
        # 构建多个 KANBlock
        self.kan_blocks = nn.ModuleList([
            KANBlock(input_dim=hidden_dim * 2, spline_order=3, grid_size=gridsize) 
            for _ in range(num_blocks)
        ])   #频
        self.kan_blocks2 = nn.ModuleList([
            KANBlock(input_dim=hidden_dim, spline_order=3, grid_size=gridsize) 
            for _ in range(num_blocks)
        ])   #空
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim*2, 3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim*2, hidden_dim*2, 3,1,1)
        )     
        self.decoder2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=1, bias=False),  # 调整通道数
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1)
        )#空

        self.conv_simple = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.hin_res_block = HinResBlock(in_size=64, out_size=64, relu_slope=0.2, use_HIN=True)
        # 添加一个新的卷积层，用于调整通道数到 64
        self.conv_to_64 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1)


    def _spatial_branch(self, x):
        """
        Updated Spatial KAN branch with kan_blocks encapsulating CS-KAN.
        Input x: [B, H, W, C]
        Output: [B, C, H, W]
        """
        B, H, W, C = x.shape

        x_ln = self.norm(x)  # LN(F_k)


        f_enhanced = self.kan_blocks[0](x_ln)  # 输出仍是 [B, H, W, C]


        q = torch.matmul(f_enhanced, self.W_q)
        k = torch.matmul(f_enhanced, self.W_k)
        v = torch.matmul(f_enhanced, self.W_v)

        q = q.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / (C // self.num_heads) ** 0.5, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, H, W, C)
        msa_out = torch.matmul(out, self.W_o)

        f_qkv = msa_out + f_enhanced  # Eq.(13)

        f_ln2 = self.norm(f_qkv)
        f_ln2 = f_ln2.permute(0, 3, 1, 2)  # -> [B, C, H, W]

        f_out = self.kan_blocks[1](f_ln2)  # CS-KAN block #2（用卷积结构）
        f_out = f_out + f_qkv.permute(0, 3, 1, 2)  # Eq.(14)

        return f_out  # [B, C, H, W]

    def _frequency_branch(self, x, h, w):
        """
        Frequency branch of SFIKAN.
        
        Args:
            x: Tensor, shape [B, H, W, C]
            h, w: int, target output spatial resolution

        Returns:
            x_reconstructed: Tensor, shape [B, C, H, W] — frequency branch output in spatial domain
        """


        x_freq = fft.rfft2(x.permute(0, 3, 1, 2))  # [B, C, H, W] → complex [B, C, H, W//2 + 1]

        x_concat = torch.cat([x_freq.real, x_freq.imag], dim=1)


        x_conv = self.freq_conv(x_concat)  # 1x1卷积, [B, 2C, H, W//2+1]

        for block in self.kan_blocks:
            x_conv = block(x_conv)


        x_processed = self.lrelu(x_conv)


        c = x_processed.shape[1] // 2
        x_complex = torch.complex(x_processed[:, :c], x_processed[:, c:])
        x_reconstructed = fft.irfft2(x_complex)  # [B, C, H, W]


        x_reconstructed = F.interpolate(x_reconstructed, size=[h, w], mode='bilinear', align_corners=False)

        return x_reconstructed  # 返回频域处理后的图像特征

        

    def _transform(self, x):
        """
        Full SFIKAN forward transform with 3-stage spatial and frequency branches.
        Input x: [B, H, W, C]
        Output: fused feature [B, C, H, W]
        """
        b, h, w, c = x.shape

        # ========== Spatial Branch ==========
        x_spatial = x
        for _ in range(3):
            x_spatial = self._spatial_branch(x_spatial)  # [B, C, H, W]

        # ========== Frequency Branch ==========
        x_frequency = x
        for _ in range(3):
            x_frequency = self._frequency_branch(x_frequency, h, w)  # [B, C, H, W]

        # ========== Fusion ==========
        x = self.decoder(torch.cat([x_spatial, x_frequency], dim=1))  # [B, 2C, H, W] -> [B, C, H, W]

        return x

    

    def _forward(self, input_x: torch.Tensor, input_b: torch.Tensor):


        x, b = input_x.permute(0, 2, 3, 1), input_b.permute(0, 2, 3, 1)

        x = self.conv_simple((torch.cat([x, b], dim=3)).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
  
        fused = self.transform(x)


        return fused


   



    def forward(self, input_x: torch.Tensor, input_b: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint(self._forward, input_x, input_b)
        else:
            return self._forward(input_x, input_b)
        



    




class KANBlock(nn.Module):
    def __init__(self, input_dim, spline_order=3, grid_size=5, act_ratio=0.125):
        super(KANBlock, self).__init__()
        reduce_channels = int(input_dim * act_ratio)

        # Channel Attention
        self.channel_fc = nn.Sequential(
            nn.Linear(input_dim, reduce_channels),
            nn.GELU()
        )
        self.channel_fc2 = nn.Sequential(
            nn.Linear(reduce_channels, input_dim),  # 全连接层，输入输出维度相同
            nn.Sigmoid()  # 使用 Sigmoid 激活函数
        )

        # Spatial Attention
        self.spatial_fc = nn.Sequential(
            nn.Linear(input_dim, reduce_channels),
            nn.GELU()
        )
        self.spatial_fc2 = nn.Sequential(
            nn.Linear(reduce_channels, input_dim),
            nn.Sigmoid()
        )

        # KAN Layers
        self.kan_layer_channel_1 = KANLinear(input_dim, input_dim, spline_order=spline_order, grid_size=grid_size)
        self.kan_layer_channel_2 = KANLinear(input_dim, input_dim, spline_order=spline_order, grid_size=grid_size)
        self.kan_layer_spatial_1 = KANLinear(input_dim, input_dim, spline_order=spline_order, grid_size=grid_size)
        self.kan_layer_spatial_2 = KANLinear(input_dim, input_dim, spline_order=spline_order, grid_size=grid_size)

    def forward(self, x):
        shortcut = x

        # Channel Attention Path
        channel_score = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)  # Compute mean along spatial dimensions [B, C, H, W] -> [B, C]
        channel_score = self.channel_fc(channel_score)  # [B, C] -> [B, reduced_channels]
        channel_score = self.channel_fc2(channel_score)  # [B, reduced_channels] -> [B, C]
        channel_score = self.kan_layer_channel_1(channel_score)  # KAN 层处理
        channel_score = self.kan_layer_channel_2(channel_score)
        channel_score = rearrange(channel_score, 'b c -> b c 1 1')  # Reshape back to [B, C, 1, 1]


        # Spatial Attention Path
        spatial_score = x.flatten(start_dim=2).mean(dim=-1)  # Flatten and mean along spatial dimensions [B, C, H*W] -> [B, C]
        spatial_score = self.spatial_fc(spatial_score)  # [B, C] -> [B, reduced_channels]
        spatial_score = self.spatial_fc2(spatial_score)  # [B, reduced_channels] -> [B, C]
        spatial_score = self.kan_layer_spatial_1(spatial_score)
        spatial_score = self.kan_layer_spatial_2(spatial_score)
        spatial_score = rearrange(spatial_score, 'b c -> b c 1 1')  # Reshape back to [B, C, 1, 1]


        # Combine Attention Paths
        x = x * channel_score + x * spatial_score



        return x + shortcut