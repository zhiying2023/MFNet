'''
功能：本文模型参数量、计算成本及推理速度的实现
FLOPs：浮点运算量，单位G
Params：参数量，可以选择将结果除以1024**2，得到以M为单位的参数量
FPS：每秒帧率
'''
import sys 
from model.mamba_vision import MambaVision
import torch
import torch.nn as nn
from collections import OrderedDict
import os
from functools import partial
import torch.nn.functional as F
from mmseg.models.decode_heads.umixformer_head import APFormerHead2
from einops import rearrange
from model.Token_Decoder import Token_Decoder


class SA(nn.Module):
    def __init__(self,channel):
        super(SA, self).__init__()
        self.conv = nn.Conv2d(channel, 1, kernel_size=1, stride=1)

    def forward(self, input):
        out = torch.sigmoid(self.conv(input))
        out = input*out

        return out

class edge(nn.Module):
    def __init__(self, channel):
        super(edge, self).__init__()
        self.conv_in = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(channel), nn.ReLU(inplace=True), SA(channel))
        self.conv_out = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(channel), nn.ReLU(inplace=True), SA(channel))
        self.linear_in= nn.Conv2d(channel, 1, kernel_size=3, padding=1)
        self.linear_out= nn.Conv2d(channel, 1, kernel_size=3, padding=1)
        self.linear_fuse= nn.Conv2d(channel*3, 1, kernel_size=3, padding=1)
        self.linear_focal= nn.Conv2d(channel, 1, kernel_size=3, padding=1)
        self.linear_rgb= nn.Conv2d(channel, 1, kernel_size=3, padding=1)
        self.linear_depth= nn.Conv2d(channel, 1, kernel_size=3, padding=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)
    def forward(self, f_rgb, f_fuse, d_focal, d_rgb, d_depth):
        f_fuse=F.interpolate(f_fuse, size=(256, 256), mode='bilinear', align_corners=False)

        f_in1=self.conv_in(f_rgb)
        f_out1=self.conv_out(f_rgb)

        f_in2 = f_fuse * torch.sigmoid(f_in1) + f_in1
        f_out2 = f_fuse * (1-torch.sigmoid(f_out1)) + f_out1

        f_in3=self.linear_in(f_in2)
        f_out3=self.linear_out(f_out2)
        f_fuse2=self.linear_fuse(torch.concat((f_fuse,f_in2,f_out2),dim=1))
        # d_focal2=self.linear_focal(torch.concat((d_focal,f_in2,f_out2),dim=1))
        # d_rgb2=self.linear_rgb(torch.concat((d_rgb,f_in2,f_out2),dim=1))
        # d_depth2=self.linear_depth(torch.concat((d_depth,f_in2,f_out2),dim=1))

        d_focal2=self.linear_focal(d_focal)
        d_rgb2=self.linear_rgb(d_rgb)
        d_depth2=self.linear_depth(d_depth)

        
        d_focal2=F.interpolate(d_focal2, size=(256, 256), mode='bilinear', align_corners=False)
        d_rgb2=F.interpolate(d_rgb2, size=(256, 256), mode='bilinear', align_corners=False)
        d_depth2=F.interpolate(d_depth2, size=(256, 256), mode='bilinear', align_corners=False)

        return f_fuse2,d_focal2,d_rgb2,d_depth2,f_in3,f_out3
    
class extract_context(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(extract_context, self).__init__()
        d = 1
        self.extra_model = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, padding=d),
                                         nn.ReLU(),
                                         nn.Conv2d(out_channel, out_channel, 3, padding=d),
                                         nn.ReLU(),
                                         nn.Conv2d(out_channel, out_channel, 3, padding=d))
    def forward(self, x):
        pred = self.extra_model(x)
        return pred

class LF(nn.Module):
    def __init__(self):
        super(LF, self).__init__()
        self.focal_encoder = MambaVision(depths=[1, 3, 8, 4],
                            num_heads=[2, 4, 8, 16],
                            window_size=[8, 8, 14, 7],
                            dim=80,
                            in_dim=32,
                            mlp_ratio=4,
                            resolution=256,
                            drop_path_rate=0.2
                            )
        self.rgb_encoder = MambaVision(depths=[1, 3, 8, 4],
                            num_heads=[2, 4, 8, 16],
                            window_size=[8, 8, 14, 7],
                            dim=80,
                            in_dim=32,
                            mlp_ratio=4,
                            resolution=256,
                            drop_path_rate=0.2
                            )
        self.depth_encoder = MambaVision(depths=[1, 3, 8, 4],
                            num_heads=[2, 4, 8, 16],
                            window_size=[8, 8, 14, 7],
                            dim=80,
                            in_dim=32,
                            mlp_ratio=4,
                            resolution=256,
                            drop_path_rate=0.2
                            )

        self.cosod_former = Token_Decoder(
                                                enc_channel=[80, 160, 320, 640],
                                                ga_channel=[320],
                                                fea_channel=[80, 160, 320, 640, 640],
                                                hidden_dim=80,
                                                num_heads=8,
                                                feedforward_dim=160,
                                                drop_path=0.1,
                                            )
        norm_cfg = dict(type='SyncBN', requires_grad=True)
        self.focal_decoder = APFormerHead2(feature_strides=[4, 8, 16, 32],
                                            in_channels=[80, 80, 80, 80],
                                            in_index=[0, 1, 2, 3],
                                            channels=320,
                                            dropout_ratio=0.1,
                                            decoder_params=dict(embed_dim=160,
                                                                num_heads=[8, 5, 2, 1],
                                                                pool_ratio=[1, 2, 4, 8]),
                                            num_classes=80,
                                            norm_cfg=norm_cfg,
                                            align_corners=False,
                                            loss_decode=dict(
                                                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
                                            )
        self.rgb_decoder = APFormerHead2(feature_strides=[4, 8, 16, 32],
                                            in_channels=[80, 80, 80, 80],
                                            in_index=[0, 1, 2, 3],
                                            channels=320,
                                            dropout_ratio=0.1,
                                            decoder_params=dict(embed_dim=160,
                                                                num_heads=[8, 5, 2, 1],
                                                                pool_ratio=[1, 2, 4, 8]),
                                            num_classes=80,
                                            norm_cfg=norm_cfg,
                                            align_corners=False,
                                            loss_decode=dict(
                                                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
                                            )
        self.depth_decoder = APFormerHead2(feature_strides=[4, 8, 16, 32],
                                            in_channels=[80, 80, 80, 80],
                                            in_index=[0, 1, 2, 3],
                                            channels=320,
                                            dropout_ratio=0.1,
                                            decoder_params=dict(embed_dim=160,
                                                                num_heads=[8, 5, 2, 1],
                                                                pool_ratio=[1, 2, 4, 8]),
                                            num_classes=80,
                                            norm_cfg=norm_cfg,
                                            align_corners=False,
                                            loss_decode=dict(
                                                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
                                            )
                                            
        self.focal_conv=nn.Sequential(nn.Conv2d(960, 80, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(80),
                                        nn.ReLU(inplace=True))
        self.fuse_conv=nn.Sequential(nn.Conv2d(80*3, 80, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(80),
                                        nn.ReLU(inplace=True))
        self.extract_context=extract_context(3,80)
        self.edge=edge(80)
        


    def forward(self,focal,rgb,depth):
        outs_focal = self.focal_encoder(focal)    #[12, 80, 64, 64],[12, 160, 32, 32],[12, 320, 16, 16],[12, 640, 8, 8]
        outs_rgb = self.rgb_encoder(rgb)
        outs_depth = self.depth_encoder(depth)
        outs=[]
        for i in range(len(outs_focal)):
            b,c,h,w=outs_focal[i].shape
            temp=torch.concat((outs_focal[i].reshape(-1,12,c,h,w),outs_rgb[i].reshape(-1,1,c,h,w),outs_depth[i].reshape(-1,1,c,h,w)),dim=1)
            outs.append(rearrange(temp,"ba n c h w->(ba n) c h w"))
        outputs_focal,outputs_rgb,outputs_depth,last_fea=self.cosod_former(outs)
        y1 = self.focal_decoder(outputs_focal)
        y1=rearrange(y1, "(ba b) c h w -> ba (b c) h w",b=12)
        yy1=self.focal_conv(y1)
        y2 = self.rgb_decoder(outputs_rgb)
        y3 = self.depth_decoder(outputs_depth)
        fuse=self.fuse_conv(torch.concat((yy1,y2,y3),dim=1))
        context_features = self.extract_context(rgb)
        
        fuse,pred1,pred2,pred3,pred_in,pred_ex=self.edge(context_features,fuse,yy1,y2,y3)
        return fuse,pred1,pred2,pred3,pred_in,pred_ex

if __name__ == "__main__":
    size=256
    ba=1
    a = torch.rand(12*ba,3,size,size).cuda()
    b = torch.rand(1*ba,3,size,size).cuda()
    c = torch.rand(1*ba,3,size,size).cuda()
    model = LF().cuda()
    x = model(a,b,c)
    for i in range(len(x)):
        print(x[i].shape)
    
    #每秒帧数
    model.eval()
    with torch.no_grad():
        import time
        nums = 640+462+155  #DUTLF-FS462,HUNT155,Lytro-Illum640
        time_s = time.time()
        for i in range(nums):
            _ = model(a, b, c)
        time_e = time.time()
        fps = nums / (time_e - time_s)
        print("FPS: %f" % fps)  #13.3

    from thop import profile, clever_format
    flops, params = profile(model, inputs = (a,b,c ))
    flops, parsms = clever_format([flops, params], '%.3f')
    print('FLOPs:{} Params:{}'.format(flops,params)) # 浮点运算量226.967G，参数量98151200.0/(1024**2)=93.60
    def print_layer_flops(model, focal_input, rgb_input, depth_input):
        # 计算 focal_encoder 部分的 FLOPs
        flops, params = profile(model.focal_encoder, inputs=(focal_input,))
        flops, params = clever_format([flops, params], "%.3f")
        print(f"Layer: focal_encoder, FLOPs: {flops}, Params: {params}")

        # 计算 rgb_encoder 部分的 FLOPs
        flops, params = profile(model.rgb_encoder, inputs=(rgb_input,))
        flops, params = clever_format([flops, params], "%.3f")
        print(f"Layer: rgb_encoder, FLOPs: {flops}, Params: {params}")

        # 计算 depth_encoder 部分的 FLOPs
        flops, params = profile(model.depth_encoder, inputs=(depth_input,))
        flops, params = clever_format([flops, params], "%.3f")
        print(f"Layer: depth_encoder, FLOPs: {flops}, Params: {params}")

        # 获取三个 encoder 的输出
        with torch.no_grad():
            outs_focal = model.focal_encoder(focal_input)
            outs_rgb = model.rgb_encoder(rgb_input)
            outs_depth = model.depth_encoder(depth_input)

            outs = []
            for i in range(len(outs_focal)):
                b,c,h,w=outs_focal[i].shape
                temp=torch.concat((outs_focal[i].reshape(-1,12,c,h,w), outs_rgb[i].reshape(-1,1,c,h,w), outs_depth[i].reshape(-1,1,c,h,w)), dim=1)
                outs.append(temp.reshape(-1, c, h, w))

        # 计算 cosod_former 部分的 FLOPs
        flops, params = profile(model.cosod_former, inputs=(outs,))
        flops, params = clever_format([flops, params], "%.3f")
        print(f"Layer: cosod_former, FLOPs: {flops}, Params: {params}")

        # 获取 decoder 输入
        with torch.no_grad():
            outputs_focal, outputs_rgb, outputs_depth, last_fea = model.cosod_former(outs)

        # 计算 focal_decoder
        flops, params = profile(model.focal_decoder, inputs=(outputs_focal,))
        flops, params = clever_format([flops, params], "%.3f")
        print(f"Layer: focal_decoder, FLOPs: {flops}, Params: {params}")

        # 计算 rgb_decoder
        flops, params = profile(model.rgb_decoder, inputs=(outputs_rgb,))
        flops, params = clever_format([flops, params], "%.3f")
        print(f"Layer: rgb_decoder, FLOPs: {flops}, Params: {params}")

        # 计算 depth_decoder
        flops, params = profile(model.depth_decoder, inputs=(outputs_depth,))
        flops, params = clever_format([flops, params], "%.3f")
        print(f"Layer: depth_decoder, FLOPs: {flops}, Params: {params}")

        # 构造融合后的 tensor 用于 fuse_conv
        with torch.no_grad():
            y1 = model.focal_decoder(outputs_focal)
            y1 = y1.reshape(-1, 960, y1.shape[2], y1.shape[3])  # 假设 focal_decoder 输出的是 [ba*b, 80, h, w]
            yy1 = model.focal_conv(y1)
            y2 = model.rgb_decoder(outputs_rgb)
            y3 = model.depth_decoder(outputs_depth)
            fuse_input = torch.concat((yy1, y2, y3), dim=1)

        # 计算 fuse_conv
        flops, params = profile(model.fuse_conv, inputs=(fuse_input,))
        flops, params = clever_format([flops, params], "%.3f")
        print(f"Layer: fuse_conv, FLOPs: {flops}, Params: {params}")

        # 计算 extract_context
        flops, params = profile(model.extract_context, inputs=(rgb_input,))
        flops, params = clever_format([flops, params], "%.3f")
        print(f"Layer: extract_context, FLOPs: {flops}, Params: {params}")

        # 计算 edge
        context_features = model.extract_context(rgb_input)
        fuse = model.fuse_conv(torch.cat((yy1, y2, y3), dim=1))  # 保证是 80 通道
        flops, params = profile(model.edge, inputs=(context_features, fuse, yy1, y2, y3))
        flops, params = clever_format([flops, params], "%.3f")
        print(f"Layer: edge, FLOPs: {flops}, Params: {params}")

    # 构造模型和测试输入
    model = LF().cuda()
    # 调用 FLOPs 打印函数
    print_layer_flops(model, a,b,c)



    