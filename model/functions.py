import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn

def AP(img): # N C H W 
    return torch.mean(img, dim=1, keepdim=True) 

def images_gradient(images):
    ret = torch.abs(images[:,:,:-1,:-1] - images[:,:,1:,:-1]) + torch.abs(images[:,:,:-1,:-1] - images[:,:,:-1,1:])
    return ret

def get_edge(data):
    rs = F.avg_pool2d(data, kernel_size=5, stride=1, padding=2)
    rs = data - rs
    return rs

def get_highpass(data):
    Tensor = torch.cuda.FloatTensor
    kernel = [[1.,1.,1.],[1.,-8.,1.],[1.,1.,1.]]
    min_batch, channels = data.size()[:2]
    kernel = Tensor(kernel).expand(channels, channels, 3 ,3)
    weight = nn.Parameter(data=kernel, requires_grad=False)

    return F.conv2d(data, weight, stride=1, padding=1)

def trim_image(image, L = 0, R = 2**11):
        L = torch.Tensor([L]).float().cuda()
        R = torch.Tensor([R]).float().cuda()
        return torch.min(torch.max(image, L), R)

def gradient_penalty(cuda, D, real_samples, fake_samples):
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = autograd.Variable(Tensor(d_interpolates.size()).fill_(1.0), requires_grad=False)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1) # 展开成1维
    gradient_penalty = ( (gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

def repeat(imgs, r=4):
    return torch.cat([imgs for _ in range(r)], dim=1)

def upsample(imgs, r=4, mode='bicubic'):
    _, __, h, w = imgs.shape
    return F.interpolate(imgs, size=[h * r, w * r], mode=mode, align_corners=True)

def downsample(imgs, r=4, mode='bicubic'):
    _, __, h, w = imgs.shape
    return F.interpolate(imgs, size=[h // r, w // r], mode=mode, align_corners=True)

def blursample(imgs, r=4):
    return upsample( downsample(imgs, r), r)

def weight_init(m):
    # for m in self.modules():
    #    if isinstance(m, nn.Conv2d):
    #        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #        m.weight.data.normal_(0, math.sqrt(2. / n))
    #        if m.bias is not None:
    #            m.bias.data.zero_()
    #    elif isinstance(m, nn.BatchNorm2d):
    #        m.weight.data.fill_(1)
    #        if m.bias is not None:
    #            m.bias.data.zero_()
    #    print (m)
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.zero_()

def get_gradient(img):
    img_xy = torch.gradient(img, axis=(2, 3))
    img_x = img_xy[0]
    img_y = img_xy[1]
    # grad_pan_fuse_ref = torch.cat([pan_x,pan_y],1)
    grad_img_fuse_ref = torch.sqrt(img_x ** 2 + img_y ** 2)
    return grad_img_fuse_ref
        

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape)==4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)


def to_2d(x):
    return rearrange(x, 'b c h w -> (b h w) c')

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)