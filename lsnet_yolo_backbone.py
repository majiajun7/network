"""
LSNet backbone for YOLOv8 pose model
Based on the paper: LSNet: Light-weight and Scalable Network
"""

import torch
import torch.nn as nn
from typing import List, Optional
import itertools
from timm.models.layers import SqueezeExcite


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, 
            dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class RepVGGDW(nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
    
    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x
    
    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()
        
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        
        conv1_w = nn.functional.pad(conv1_w, [1,1,1,1])
        identity = nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [1,1,1,1])
        
        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b
        
        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv


class SKA(nn.Module):
    """Simplified SKA module without triton dependency"""
    def __init__(self, dim, sks=3, groups=8):
        super().__init__()
        self.dim = dim
        self.sks = sks
        self.groups = groups
        
        # Dynamic kernel generation
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=sks, stride=1, 
                               padding=(sks-1)//2, groups=dim, bias=False)
        
    def forward(self, x, w):
        # Simplified version: apply dynamic weights as channel-wise attention
        b, c, h, width = x.shape
        # Average pooling over spatial kernel dimensions
        w = w.mean(dim=2)  # [b, c//groups, h, w]
        
        # Expand to match input channels
        w = w.repeat(1, self.groups, 1, 1)[:, :c, :, :]
        
        # Apply depthwise convolution with attention
        x = self.dw_conv(x * w)
        return x


class LKP(nn.Module):
    """Large Kernel Perception module"""
    def __init__(self, dim, lks=7, sks=3, groups=8):
        super().__init__()
        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = nn.ReLU()
        self.cv2 = Conv2d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv2d_BN(dim // 2, dim // 2)
        self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)
        
        self.sks = sks
        self.groups = groups
        self.dim = dim
        
    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))
        b, _, h, width = w.size()
        w = w.view(b, self.dim // self.groups, self.sks ** 2, h, width)
        return w


class LSConv(nn.Module):
    """Large-Small Convolution block"""
    def __init__(self, dim):
        super(LSConv, self).__init__()
        self.lkp = LKP(dim, lks=7, sks=3, groups=8)
        self.ska = SKA(dim)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(self.ska(x, self.lkp(x))) + x


class FFN(nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


class Residual(nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class Attention(nn.Module):
    """Simplified attention module for LSNet"""
    def __init__(self, dim, key_dim, num_heads=8, attn_ratio=4, resolution=14):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        
        self.qkv = Conv2d_BN(dim, h, ks=1)
        self.proj = nn.Sequential(nn.ReLU(), Conv2d_BN(self.dh, dim, bn_weight_init=0))
        self.dw = Conv2d_BN(nh_kd, nh_kd, 3, 1, 1, groups=nh_kd)
        
        # Simplified relative position bias
        self.resolution = resolution
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, resolution, resolution))

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, -1, H, W).split([self.nh_kd, self.nh_kd, self.dh], dim=1)
        q = self.dw(q)
        q, k, v = q.view(B, self.num_heads, -1, N), k.view(B, self.num_heads, -1, N), v.view(B, self.num_heads, -1, N)
        
        attn = (q.transpose(-2, -1) @ k) * self.scale
        
        # Add position bias if resolution matches
        if H == self.resolution and W == self.resolution:
            attn = attn + self.attention_biases.unsqueeze(0).view(1, self.num_heads, N, N)
        
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).reshape(B, -1, H, W)
        x = self.proj(x)
        return x


class LSNetBlock(nn.Module):
    """LSNet block with mixed architecture"""
    def __init__(self, ed, kd, nh=8, ar=4, resolution=14, stage=-1, depth=-1):
        super().__init__()
            
        if depth % 2 == 0:
            self.mixer = RepVGGDW(ed)
            self.se = SqueezeExcite(ed, 0.25)
        else:
            self.se = nn.Identity()
            if stage == 3:  # Last stage uses attention
                self.mixer = Residual(Attention(ed, kd, nh, ar, resolution=resolution))
            else:
                self.mixer = LSConv(ed)

        self.ffn = Residual(FFN(ed, int(ed * 2)))

    def forward(self, x):
        return self.ffn(self.se(self.mixer(x)))


class LSNetBackbone(nn.Module):
    """
    LSNet backbone for YOLOv8
    Returns features at different scales for FPN
    """
    def __init__(self, 
                 variant='lsnet_t',
                 in_chans=3,
                 out_features=None):
        super().__init__()
        
        # Model configurations
        configs = {
            'lsnet_t': {
                'embed_dim': [64, 128, 256, 384],
                'depth': [0, 2, 8, 10],
                'num_heads': [3, 3, 3, 4],
                'key_dim': [16, 16, 16, 16],
            },
            'lsnet_s': {
                'embed_dim': [96, 192, 320, 448],
                'depth': [1, 2, 8, 10],
                'num_heads': [3, 3, 3, 4],
                'key_dim': [16, 16, 16, 16],
            },
            'lsnet_b': {
                'embed_dim': [128, 256, 384, 512],
                'depth': [4, 6, 8, 10],
                'num_heads': [3, 3, 3, 4],
                'key_dim': [16, 16, 16, 16],
            }
        }
        
        cfg = configs[variant]
        self.embed_dim = cfg['embed_dim']
        self.depth = cfg['depth']
        self.num_heads = cfg['num_heads']
        self.key_dim = cfg['key_dim']
        
        # Stem - outputs 1/8 resolution
        self.stem = nn.Sequential(
            Conv2d_BN(in_chans, self.embed_dim[0] // 4, 3, 2, 1),
            nn.ReLU(),
            Conv2d_BN(self.embed_dim[0] // 4, self.embed_dim[0] // 2, 3, 2, 1),
            nn.ReLU(),
            Conv2d_BN(self.embed_dim[0] // 2, self.embed_dim[0], 3, 2, 1)
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        resolution = 28  # Assuming 224 input, after stem it's 28x28
        
        for i in range(4):
            stage = nn.Sequential()
            
            # Add blocks for this stage
            for d in range(self.depth[i]):
                attn_ratio = self.embed_dim[i] / (self.key_dim[i] * self.num_heads[i])
                stage.append(LSNetBlock(
                    self.embed_dim[i], self.key_dim[i], self.num_heads[i], 
                    attn_ratio, resolution, stage=i, depth=d
                ))
            
            # Add downsampling if not last stage
            if i < 3:
                stage.append(Conv2d_BN(self.embed_dim[i], self.embed_dim[i], ks=3, stride=2, pad=1, groups=self.embed_dim[i]))
                stage.append(Conv2d_BN(self.embed_dim[i], self.embed_dim[i+1], ks=1, stride=1, pad=0))
                resolution = (resolution - 1) // 2 + 1
            
            self.stages.append(stage)
        
        # Output features
        self.out_features = out_features or ["stage1", "stage2", "stage3"]
        self._out_feature_channels = {
            "stage0": self.embed_dim[0],
            "stage1": self.embed_dim[1], 
            "stage2": self.embed_dim[2],
            "stage3": self.embed_dim[3]
        }
        self._out_feature_strides = {
            "stage0": 8,
            "stage1": 16,
            "stage2": 32, 
            "stage3": 64
        }
        
    def forward(self, x):
        """
        Returns dict of features at different scales
        """
        features = {}
        
        x = self.stem(x)
        
        for i, stage in enumerate(self.stages):
            x = stage(x)
            stage_name = f"stage{i}"
            if stage_name in self.out_features:
                features[stage_name] = x
        
        return features
    
    def output_shape(self):
        return {
            name: {
                "channels": self._out_feature_channels[name],
                "stride": self._out_feature_strides[name]
            }
            for name in self.out_features
        }


# YOLOv8 compatible interface
class LSNetYOLOBackbone(nn.Module):
    """
    LSNet backbone wrapper for YOLOv8 integration
    """
    def __init__(self, variant='lsnet_t', channels=[128, 256, 512], pretrained=False):
        super().__init__()
        
        self.backbone = LSNetBackbone(variant=variant)
        
        # Get actual output channels from LSNet
        lsnet_channels = [
            self.backbone._out_feature_channels[f"stage{i}"] 
            for i in range(1, 4)
        ]
        
        # Channel adaptation layers if needed
        self.adapters = nn.ModuleList()
        for i, (lsnet_ch, yolo_ch) in enumerate(zip(lsnet_channels, channels)):
            if lsnet_ch != yolo_ch:
                adapter = Conv2d_BN(lsnet_ch, yolo_ch, 1, 1, 0)
            else:
                adapter = nn.Identity()
            self.adapters.append(adapter)
        
        self.channels = channels
        
    def forward(self, x):
        """
        Returns list of features for YOLOv8 neck
        """
        features = self.backbone(x)
        
        # Extract and adapt features for YOLO
        outputs = []
        for i, adapter in enumerate(self.adapters):
            feat = features[f"stage{i+1}"]
            outputs.append(adapter(feat))
        
        return outputs
    
    @property
    def stride(self):
        """Return stride of each feature level"""
        return [16, 32, 64]  # P3, P4, P5


def create_lsnet_backbone(variant='lsnet_t', channels=[128, 256, 512], pretrained=False):
    """
    Factory function to create LSNet backbone for YOLOv8
    
    Args:
        variant: 'lsnet_t', 'lsnet_s', or 'lsnet_b'
        channels: output channels for YOLOv8 neck
        pretrained: whether to load pretrained weights
    """
    return LSNetYOLOBackbone(variant, channels, pretrained)