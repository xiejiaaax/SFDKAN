
import torch
import torch.nn as nn
import torch.nn.functional as F
from .moe import HinResBlock,FuseNet3
from .vmamba_efficient import Fourier_VSSBlock as EVSS
from .FreqNCELoss import FreqNCELoss, cc_loss, pixel_cc_loss
from wtkan.wtkan2d import WTKan2d
import os

class Net_1EVSS(nn.Module):
    def __init__(self, num_channels=1, base_filter=32, num_experts=4, topk=2):
        super(Net_1EVSS, self).__init__()
        self.base_filter = base_filter
        self.stride=1
        self.patch_size=1
        self.embed_dim = base_filter*self.stride*self.patch_size
        # self.shared_encoder = nn.Sequential(nn.Conv2d(num_channels,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
        self.vis_encoder = nn.Sequential(nn.Conv2d(num_channels,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
        self.ir_encoder = nn.Sequential(nn.Conv2d(num_channels,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
        self.vis_feature_extraction = nn.Sequential(*[EVSS_Block(self.embed_dim) for i in range(1)])
        self.ir_feature_extraction = nn.Sequential(*[EVSS_Block(self.embed_dim) for i in range(1)])
        self.vis_decoder = InvSplit(self.embed_dim*2, self.embed_dim)
        self.ir_decoder = InvSplit(self.embed_dim*2, self.embed_dim)
        
        self.cross_fusion = FuseNet(channels=base_filter, num_experts=num_experts, k=topk)
        self.fuse_decoder = InvSplit(self.embed_dim*2, self.embed_dim)
        self.decoder = nn.Conv2d(base_filter,num_channels,kernel_size=1)

        self.vis_pixel_decoder = InvSplit(4, 2, is_pixel=True)
        self.ir_pixel_decoder = InvSplit(4, 2, is_pixel=True)
        self.fuse_pixel_decoder = InvSplit(4, 2, is_pixel=True)
        self.conv_b = nn.Conv2d(4,2,1)
        self.conv_d = nn.Conv2d(4,2,1)


    def Count_Pixel_Contrast_Loss(self, fusep, irp, visp):
        ir_bp, ir_dp = self.ir_pixel_decoder(irp)
        vis_bp, vis_dp = self.vis_pixel_decoder(visp)
        fuse_bp, fuse_dp = self.fuse_pixel_decoder(fusep)
        fuse_bp_hat = self.conv_b(torch.cat((ir_bp, vis_bp),dim=1))
        fuse_dp_hat = self.conv_b(torch.cat((ir_dp, vis_dp),dim=1))
        loss_pixel_cc = pixel_cc_loss(fuse_bp, fuse_dp, fuse_bp_hat, fuse_dp_hat)
        return loss_pixel_cc

    def Count_Feature_Contrast_Loss(self, fusef, irf, visf, stage=1):
        ir_b, ir_d = self.ir_decoder(irf)
        vis_b, vis_d = self.vis_decoder(visf)
        loss_cc = cc_loss(ir_b, ir_d, vis_b, vis_d)
        if stage==1:
            return loss_cc
        elif stage==2:
            fuse_b, fuse_d = self.fuse_decoder(fusef)
            L_FreqNCE = FreqNCELoss(device=fuse_b.device)
            loss_FreqNCE = L_FreqNCE(fuse_b, fuse_d, ir_b+vis_b, ir_d+vis_d).mean()
            return loss_cc, loss_FreqNCE


    def forward(self,ir,vis,scale=1,stage=2,is_train=False,is_count_cc=False):
        ir_bic = F.interpolate(input=ir, scale_factor=scale)  
        # print('ir_bic.shape={}'.format(ir_bic.shape))
        ir_f = self.ir_feature_extraction((self.ir_encoder(ir_bic)).permute(0,2,3,1)).permute(0,3,1,2)
        # b,c,h,w = ir_f.shape
        # print('ir_f.shape={}'.format(ir_f.shape))
        vis_f = self.vis_feature_extraction((self.vis_encoder(vis)).permute(0,2,3,1)).permute(0,3,1,2)
        # ir_f = self.ir_feature_extraction((self.shared_encoder(ir_bic)).permute(0,2,3,1)).permute(0,3,1,2)
        # vis_f = self.vis_feature_extraction((self.shared_encoder(vis)).permute(0,2,3,1)).permute(0,3,1,2)
        
        if is_train:
            if stage==1:
                return self.Count_Feature_Contrast_Loss(fusef=None, irf=ir_f, visf=vis_f, stage=stage)
            elif stage==2:
                # print('ir_b.shape={}'.format(ir_b.shape), 'vis_b.shape={}'.format(vis_b.shape))
                fuse_f, cof = self.cross_fusion(ir_f, vis_f)
                fuse_img = self.decoder(fuse_f)
                # print(fuse_f.shape)
                if is_count_cc:
                    loss_pixel_cc = self.Count_Pixel_Contrast_Loss(fuse_img, ir_bic, vis)
                    loss_cc, loss_FreqNCE = self.Count_Feature_Contrast_Loss(fusef=fuse_f, irf=ir_f, visf=vis_f, stage=stage)
                    return fuse_img, cof, loss_pixel_cc, loss_cc, loss_FreqNCE
                
                return fuse_img, cof
        else:
            fuse_f = self.cross_fusion(ir_f, vis_f, infer=True)
            fuse_img = self.decoder(fuse_f)
            return fuse_img


class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)  # 恒等映射
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)  # 第一次卷积
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)  # 第二次卷积
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
            self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))  # 第一次卷积 + 激活
        if self.use_HIN:  # 如果使用 HIN
            out_1, out_2 = torch.chunk(resi, 2, dim=1)  # 分块
            resi = torch.cat([self.norm(out_1), out_2], dim=1)  # HIN处理
        resi = self.relu_2(self.conv_2(resi))  # 第二次卷积 + 激活
        return self.identity(x) + resi  # 残差连接


class Net3(nn.Module):
    def __init__(self, num_channels=1, base_filter=32, num_experts=4, topk=2):
        super(Net3, self).__init__()
        self.base_filter = base_filter
        self.stride=1
        self.patch_size=1
        self.embed_dim = base_filter*self.stride*self.patch_size

        self.visualize_path = "/home/xiejx/imagefusion/TGRS-muti-focus/data/Grad_cam"
        os.makedirs(self.visualize_path, exist_ok=True)


                # 替换卷积层为小波卷积
        # 替换MSFE为WTKan处理
        # 这里我们直接使用WTKan来替代原本的MSFE模块
        self.vis_encoder = nn.Sequential(
            nn.Conv2d(num_channels, base_filter, 1, 1, 0),
            WTKan2d(in_channels=base_filter, out_channels=base_filter, bias=True, hidden_dim=128, wt_levels=1)
        )

        self.ir_encoder = nn.Sequential(
            nn.Conv2d(num_channels, base_filter, 1, 1, 0),
            WTKan2d(in_channels=base_filter, out_channels=base_filter, bias=True, hidden_dim=128, wt_levels=1)
        )
        # self.vis_encoder = nn.Sequential(
        #             HinResBlock(in_size=num_channels, out_size=base_filter, relu_slope=0.2, use_HIN=True)  # 替换两层为一个 HinResBlock
        #             )

        # self.ir_encoder = nn.Sequential(
        #             HinResBlock(in_size=num_channels, out_size=base_filter, relu_slope=0.2, use_HIN=True)  # 替换两层为一个 HinResBlock
        #             )

        #self.ir_expand = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1)
        # self.shared_encoder = nn.Sequential(HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
        #self.vis_encoder = nn.Sequential(nn.Conv2d(num_channels,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
        #self.ir_encoder = nn.Sequential(nn.Conv2d(num_channels,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
        
        
        # self.vis_feature_extraction = nn.Sequential(*[EVSS_Block(self.embed_dim) for i in range(2)])
        # self.ir_feature_extraction = nn.Sequential(*[EVSS_Block(self.embed_dim) for i in range(2)])
        self.vis_decoder = InvSplit(self.embed_dim*2, self.embed_dim)
        self.ir_decoder = InvSplit(self.embed_dim*2, self.embed_dim)
        
        self.cross_fusion = FuseNet3(var_num=base_filter, num_experts=4)
        self.fuse_decoder = InvSplit(self.embed_dim*2, self.embed_dim)
        self.decoder = nn.Conv2d(base_filter*2,num_channels,kernel_size=1)

        self.vis_pixel_decoder = InvSplit(4, 2, is_pixel=True)
        self.ir_pixel_decoder = InvSplit(4, 2, is_pixel=True)
        self.fuse_pixel_decoder = InvSplit(4, 2, is_pixel=True)
        self.conv_b = nn.Conv2d(4,2,1)
        self.conv_d = nn.Conv2d(4,2,1)


    def Count_Pixel_Contrast_Loss(self, fusep, irp, visp):
        ir_bp, ir_dp = self.ir_pixel_decoder(irp)
        vis_bp, vis_dp = self.vis_pixel_decoder(visp)
        fuse_bp, fuse_dp = self.fuse_pixel_decoder(fusep)
        fuse_bp_hat = self.conv_b(torch.cat((ir_bp, vis_bp),dim=1))
        fuse_dp_hat = self.conv_b(torch.cat((ir_dp, vis_dp),dim=1))
        loss_pixel_cc = pixel_cc_loss(fuse_bp, fuse_dp, fuse_bp_hat, fuse_dp_hat)
        return loss_pixel_cc

    def Count_Feature_Contrast_Loss(self, fusef, irf, visf, stage=1):
        ir_b, ir_d = self.ir_decoder(irf)
        vis_b, vis_d = self.vis_decoder(visf)
        loss_cc = cc_loss(ir_b, ir_d, vis_b, vis_d)
        if stage==1:
            return loss_cc
        elif stage==2:
            fuse_b, fuse_d = self.fuse_decoder(fusef)
            L_FreqNCE = FreqNCELoss(device=fuse_b.device)
            loss_FreqNCE = L_FreqNCE(fuse_b, fuse_d, ir_b+vis_b, ir_d+vis_d).mean()
            return loss_cc, loss_FreqNCE


    def forward(self,ir,vis,scale=1):
        ir_bic = F.interpolate(input=ir, scale_factor=scale)  
        #print('ir_bic.shape={}'.format(ir_bic.shape))
        # 如果ir_bic的通道数为1，复制通道以匹配模型的预期
        #print("ir",ir.shape)
        ir_f = self.ir_encoder(ir_bic).permute(0,2,3,1).permute(0,3,1,2)



        # b,c,h,w = ir_f.shape
        # print('ir_f.shape={}'.format(ir_f.shape))
        vis_f = self.vis_encoder(vis).permute(0,2,3,1).permute(0,3,1,2)
        #print("vis_f",vis_f.shape)
        # ir_f = self.ir_feature_extraction((self.shared_encoder(ir_bic)).permute(0,2,3,1)).permute(0,3,1,2)
        # vis_f = self.vis_feature_extraction((self.shared_encoder(vis)).permute(0,2,3,1)).permute(0,3,1,2)
        

        fuse_f = self.cross_fusion(ir_f, vis_f)
        fuse_img = self.decoder(fuse_f)
                
        return fuse_img
        


    def __init__(self, num_channels=1, base_filter=32, num_experts=4, topk=2):
        super(Net_1EVSS, self).__init__()
        self.base_filter = base_filter
        self.stride=1
        self.patch_size=1
        self.embed_dim = base_filter*self.stride*self.patch_size
        # self.shared_encoder = nn.Sequential(nn.Conv2d(num_channels,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
        self.vis_encoder = nn.Sequential(nn.Conv2d(num_channels,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
        self.ir_encoder = nn.Sequential(nn.Conv2d(num_channels,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
        self.vis_feature_extraction = nn.Sequential(*[EVSS_Block(self.embed_dim) for i in range(1)])
        self.ir_feature_extraction = nn.Sequential(*[EVSS_Block(self.embed_dim) for i in range(1)])
        self.vis_decoder = InvSplit(self.embed_dim*2, self.embed_dim)
        self.ir_decoder = InvSplit(self.embed_dim*2, self.embed_dim)
        
        self.fuse_decoder = InvSplit(self.embed_dim*2, self.embed_dim)
        self.decoder = nn.Conv2d(base_filter,num_channels,kernel_size=1)

        self.vis_pixel_decoder = InvSplit(4, 2, is_pixel=True)
        self.ir_pixel_decoder = InvSplit(4, 2, is_pixel=True)
        self.fuse_pixel_decoder = InvSplit(4, 2, is_pixel=True)
        self.conv_b = nn.Conv2d(4,2,1)
        self.conv_d = nn.Conv2d(4,2,1)


    def Count_Pixel_Contrast_Loss(self, fusep, irp, visp):
        ir_bp, ir_dp = self.ir_pixel_decoder(irp)
        vis_bp, vis_dp = self.vis_pixel_decoder(visp)
        fuse_bp, fuse_dp = self.fuse_pixel_decoder(fusep)
        fuse_bp_hat = self.conv_b(torch.cat((ir_bp, vis_bp),dim=1))
        fuse_dp_hat = self.conv_b(torch.cat((ir_dp, vis_dp),dim=1))
        loss_pixel_cc = pixel_cc_loss(fuse_bp, fuse_dp, fuse_bp_hat, fuse_dp_hat)
        return loss_pixel_cc

    def Count_Feature_Contrast_Loss(self, fusef, irf, visf, stage=1):
        ir_b, ir_d = self.ir_decoder(irf)
        vis_b, vis_d = self.vis_decoder(visf)
        loss_cc = cc_loss(ir_b, ir_d, vis_b, vis_d)
        if stage==1:
            return loss_cc
        elif stage==2:
            fuse_b, fuse_d = self.fuse_decoder(fusef)
            L_FreqNCE = FreqNCELoss(device=fuse_b.device)
            loss_FreqNCE = L_FreqNCE(fuse_b, fuse_d, ir_b+vis_b, ir_d+vis_d).mean()
            return loss_cc, loss_FreqNCE


    def forward(self,ir,vis,scale=1,stage=2,is_train=False,is_count_cc=False):
        ir_bic = F.interpolate(input=ir, scale_factor=scale)  
        # print('ir_bic.shape={}'.format(ir_bic.shape))
        ir_f = self.ir_feature_extraction((self.ir_encoder(ir_bic)).permute(0,2,3,1)).permute(0,3,1,2)
        # b,c,h,w = ir_f.shape
        # print('ir_f.shape={}'.format(ir_f.shape))
        vis_f = self.vis_feature_extraction((self.vis_encoder(vis)).permute(0,2,3,1)).permute(0,3,1,2)
        # ir_f = self.ir_feature_extraction((self.shared_encoder(ir_bic)).permute(0,2,3,1)).permute(0,3,1,2)
        # vis_f = self.vis_feature_extraction((self.shared_encoder(vis)).permute(0,2,3,1)).permute(0,3,1,2)
        
        if is_train:
            if stage==1:
                return self.Count_Feature_Contrast_Loss(fusef=None, irf=ir_f, visf=vis_f, stage=stage)
            elif stage==2:
                # print('ir_b.shape={}'.format(ir_b.shape), 'vis_b.shape={}'.format(vis_b.shape))
                fuse_f, cof = self.cross_fusion(ir_f, vis_f)
                fuse_img = self.decoder(fuse_f)
                # print(fuse_f.shape)
                if is_count_cc:
                    loss_pixel_cc = self.Count_Pixel_Contrast_Loss(fuse_img, ir_bic, vis)
                    loss_cc, loss_FreqNCE = self.Count_Feature_Contrast_Loss(fusef=fuse_f, irf=ir_f, visf=vis_f, stage=stage)
                    return fuse_img, cof, loss_pixel_cc, loss_cc, loss_FreqNCE
                
                return fuse_img, cof
        else:
            fuse_f = self.cross_fusion(ir_f, vis_f, infer=True)
            fuse_img = self.decoder(fuse_f)
            return fuse_img

# 解耦
class InvSplit(nn.Module):
    def __init__(self, channel_num, channel_split_num, clamp=0.8, is_pixel=False):
        super(InvSplit,self).__init__()
        self.is_pixel = is_pixel
        if is_pixel:
            self.p2f = nn.Conv2d(1,2,1,1,0)
        self.encoder = nn.Conv2d(channel_split_num,channel_num,1,1,0)
        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2
        self.clamp = clamp
        self.predict = UNetConvBlock(self.split_len1, self.split_len2)
        self.update = UNetConvBlock(self.split_len1, self.split_len2)
        # self.low_out = nn.Conv2d(self.split_len1,3,1,1,0)
        # self.high_out = nn.Conv2d(self.split_len1,3,1,1,0)
    def forward(self,x):
        if self.is_pixel:
            x = self.p2f(x)
        x = self.encoder(x)
        # print(x.shape)
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2)) #x_1 low,x_2 high
        # print(x1.shape, x2.shape)
        x2 = x2-self.predict(x1)
        x1 = x1+self.update(x2)
        # x_low = torch.sigmoid(self.low_out(x1))
        # x_high = torch.sigmoid(self.high_out(x2))
        x_low = torch.sigmoid(x1)
        x_high = torch.sigmoid(x2)
        return x_low, x_high
        # return (x_low,x_low),(x_high,x_high)
        # return x_low, x_high


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


# 空频
class EVSS_Block(nn.Module):
    def __init__(self, MlpRatio=0, dim=32, drop_path_rate=0.1):
        super(EVSS_Block, self).__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 2)][1]
        self.encoder = EVSS(
        hidden_dim=dim, 
        drop_path=dpr,
        norm_layer=nn.LayerNorm,
        channel_first=False,
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v2",
        mlp_ratio=MlpRatio,  # 选择在Vanilla VSS模块后添加FFN层，默认值为4.0，如果不加就设为0
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        use_checkpoint=False,)

    def forward(self, x):
        return self.encoder(x)
    





