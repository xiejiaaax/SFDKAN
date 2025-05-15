from packaging import version
import torch
from torch import nn
from math import *
import torch.nn.functional as F
import torch.fft


class FreqNCELoss(nn.Module):
    def __init__(self, device='cuda:0'):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.device = device

    def forward(self, fuse_b, fuse_d, base_f, detail_f):

        batchSize = fuse_b.shape[0]
        dim = fuse_b.shape[1]
        T = 0.07

        fuse_b, fuse_d = F.normalize(fuse_b, dim=1), F.normalize(fuse_d, dim=1)
        base_f, detail_f = F.normalize(base_f, dim=1), F.normalize(detail_f, dim=1)

        # print('feat_fl.shape={}, feat_fh.shape={}, feat_vl.shape={}, feat_vh.shape={}, feat_il.shape={}, feat_ih.shape={}'.format(feat_fl.shape, feat_fh.shape, feat_vl.shape, feat_vh.shape, feat_il.shape, feat_ih.shape))

        # 
        l_pos_d = torch.bmm(fuse_d.view(batchSize, 1, -1), detail_f.view(batchSize, -1, 1))
        l_pos_d = l_pos_d.view(1,1)
        l_pos_b = torch.bmm(fuse_b.view(batchSize, 1, -1), base_f.view(batchSize, -1, 1))
        l_pos_b = l_pos_b.view(1,1)
        l_neg_d = torch.bmm(fuse_d.view(batchSize, 1, -1), base_f.view(batchSize, -1, 1))
        l_neg_d = l_neg_d.view(1,1)
        l_neg_b = torch.bmm(fuse_b.view(batchSize, 1, -1), detail_f.view(batchSize, -1, 1))
        l_neg_b = l_neg_b.view(1,1)
        
        l_pos_f = torch.zeros((2,1)).to(self.device)
        l_neg_f = torch.zeros((2,1)).to(self.device)
        
        l_pos_f = l_pos_f.scatter_add_(0, torch.tensor([[0]]).to(self.device), l_pos_d)
        l_pos_f = l_pos_f.scatter_add_(0, torch.tensor([[1]]).to(self.device), l_pos_b)
        l_neg_f = l_neg_f.scatter_add_(0, torch.tensor([[0]]).to(self.device), l_neg_d)
        l_neg_f = l_neg_f.scatter_add_(0, torch.tensor([[1]]).to(self.device), l_neg_b)

        # print('l_pos_f.shape={}, l_neg_f.shape={}'.format(l_pos_f.shape, l_neg_f.shape))
        out = torch.cat((l_pos_f, l_neg_f), dim=0)/T
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=fuse_b.device))

        return loss


def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1)/(eps + torch.sqrt(torch.sum(img1**2, dim=-1))*torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()


def cc_loss(ir_b, ir_d, vis_b, vis_d):
    cc_loss_b = cc(ir_b, vis_b)
    cc_loss_d = cc(ir_d, vis_d)
    return (cc_loss_d)**2/(1.01 + cc_loss_b)


def pixel_cc_loss(fuse_bp, fuse_dp, fuse_bp_hat, fuse_dp_hat):
    # pos
    cc_bb = cc(fuse_bp, fuse_bp_hat)
    cc_dd = cc(fuse_dp, fuse_dp_hat)
    # neg
    cc_bd = cc(fuse_bp, fuse_dp_hat)
    cc_db = cc(fuse_dp, fuse_bp_hat)
    return (cc_bd)**2/(1.01 + cc_bb) + (cc_db)**2/(1.01 + cc_dd)
