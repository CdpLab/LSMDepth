from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from layers import *
from timm.models.layers import trunc_normal_
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class ChannelAttention(nn.Module):
    """通道注意力模块（保持原结构新增）"""
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class Conv1x1(nn.Module):
    """1x1卷积层（带可选的激活函数）"""
    def __init__(self, in_channels, out_channels, use_relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        if use_relu:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = None
        
    def forward(self, x):
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
    
class SpatialAttention(nn.Module):
    """空间注意力模块（新增但保持层结构）"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_att = torch.cat([avg_out, max_out], dim=1)
        x_att = self.conv(x_att)
        return x * self.sigmoid(x_att)

class DeformConvBlock(nn.Module):
    """替代原可变形卷积模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 替换为常规卷积
        self.conv = nn.Sequential(
            Conv3x3(in_channels, out_channels),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super().__init__()
        
        # 保持原参数不变
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'bilinear'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = [ch//2 for ch in num_ch_enc]  # 修改通道缩减策略

        # 新增模块（保持原有convs结构）
        self.skip_att = nn.ModuleDict()  # 跳跃连接注意力
        self.skip_conv = nn.ModuleDict() # 通道调整
        
        # 初始化跳跃连接处理
        for i in range(3):
            self.skip_att[str(i)] = nn.Sequential(
                ChannelAttention(num_ch_enc[i]),
                SpatialAttention()
            )
            self.skip_conv[str(i)] = Conv1x1(num_ch_enc[i], self.num_ch_dec[i])

        # 修改原有解码结构
        self.convs = OrderedDict()
        for i in range(2, -1, -1):
            # upconv_0替换为可变形卷积
            num_ch_in = self.num_ch_enc[-1] if i == 2 else self.num_ch_dec[i + 1]
            self.convs[("upconv", i, 0)] = DeformConvBlock(num_ch_in, self.num_ch_dec[i])
            
            # upconv_1增加注意力
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_dec[i-1]  # 使用调整后的通道数
            self.convs[("upconv", i, 1)] = nn.Sequential(
                ConvBlock(num_ch_in, self.num_ch_dec[i]),
                ChannelAttention(self.num_ch_dec[i])
            )

        # 增强输出层
        for s in self.scales:
            self.convs[("dispconv", s)] = nn.Sequential(
                Conv3x3(self.num_ch_dec[s], 64),
                nn.ReLU(inplace=True),
                Conv3x3(64, self.num_output_channels)
            )

        # 保持原有初始化
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """保持原有初始化方式"""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        """保持原有数据流结构"""
        self.outputs = {}
        
        # 预处理跳跃连接
        skip_features = []
        for i in range(3):
            skip = self.skip_att[str(i)](input_features[i])
            skip = self.skip_conv[str(i)](skip)
            skip_features.append(skip)

        # 保持原有解码流程
        x = input_features[-1]
        for i in range(2, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]

            if self.use_skips and i > 0:
                x += [skip_features[i - 1]]  # 使用处理后的跳跃特征
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            if i in self.scales:
                f = upsample(self.convs[("dispconv", i)](x), mode='bilinear')
                self.outputs[("disp", i)] = self.sigmoid(f)

        return self.outputs