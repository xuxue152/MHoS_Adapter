# modified from: https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/model.py

from typing import Tuple, List
from collections import OrderedDict
import math
import functools
import pdb
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform

from configs import (
    CLIP_VIT_B16_PATH,
    CLIP_VIT_B32_PATH,
    CLIP_VIT_L14_PATH,
    DWCONV3D_DISABLE_CUDNN,
)
from transformers.models.bert.modeling_bert import BertConfig


class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        # self.ln_final = LayerNorm(config.hidden_size) #放开
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        # x = self.ln_final(x) #lm
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x


class Adapter(nn.Module):

    def __init__(self, in_channels, adapter_channels, kernel_size, T):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv3d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=tuple(x // 2 for x in kernel_size),
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        self.T = T
        self.offset_adapter = OffsetAdapter(in_channels, adapter_channels, (1, 3, 3), T)
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        offset = self.offset_adapter(x)
        T = self.T
        BT, L, C = x.size()
        B = BT // T
        Ca = self.conv.in_channels
        H = W = round(math.sqrt(L - 1))

        assert L - 1 == H * W
        x_id = x
        x = x[:, 1:, :]
        x = self.fc1(x)
        x = x.view(B, T, H, W, Ca).permute(0, 4, 1, 2, 3).contiguous()

        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        x = self.conv(x)
        torch.backends.cudnn.enabled = cudnn_enabled

        x = x.permute(0, 2, 3, 4, 1).contiguous().view(BT, L - 1, Ca)
        x = self.fc2(x) + offset
        x_id[:, 1:, :] += x
        return x_id


class Adapter1(nn.Module):

    def __init__(self, in_channels, adapter_channels, kernel_size, T):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv1d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        self.T = T
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        T = self.T
        BT, C = x.size()
        B = BT // T
        Ca = self.conv.in_channels
        x_id = x
        x = self.fc1(x)
        x = x.view(B, T, Ca).permute(0, 2, 1).contiguous().view(B, Ca, T)

        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        x = self.conv(x)
        torch.backends.cudnn.enabled = cudnn_enabled

        x = x.permute(0, 2, 1).contiguous().view(-1, Ca)
        x = self.fc2(x)
        x_id = x + x_id
        return x_id


class HS_Adapter_Earlier(nn.Module):

    def __init__(self, in_channels, adapter_channels, kernel_size, T):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv1d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        self.T = T
        self.offset_adapter = OffsetAdapter(in_channels, adapter_channels, (1, 3, 3), T)
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        offset = self.offset_adapter(x)
        T = self.T
        BT, L, C = x.size()#BT:8 C:768 L:197
        B = BT // T
        Ca = self.conv.in_channels
        x_id = x
        x = self.fc1(x)
        x = x.view(B, T, L, Ca).permute(0, 2, 3, 1).contiguous().view(B * L, Ca, T)

        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        x = self.conv(x)#左
        torch.backends.cudnn.enabled = cudnn_enabled

        x = x.permute(0, 2, 1).contiguous().view(B, L, T, Ca)
        x = x.permute(0, 2, 1, 3).contiguous().view(BT, L, Ca)
        x = self.fc2(x)
        x_id[:, 1:, :] += offset
        x_id = x + x_id

        return x_id

#lm
class HS_Adapter_Later(nn.Module):

    def __init__(self, in_channels, adapter_channels, kernel_size, T):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.hs_attention = StatisticAttentionBlock(
            in_channels,
            adapter_channels,
        )
        self.T = T
        self.adapter_channels = adapter_channels
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        T = self.T
        BT, L, C = x.size()#B128;T:8 C:768 L:50
        B = BT // T
        H = W = round(math.sqrt(L - 1))
        assert L - 1 == H * W
        x_id = x
        x = x[:, 1:, :].view(B, T, -1, C)
        former_id = [0] + [i for i in range(T)][:-1]
        x_former = x[:, former_id]
        offset = x - x_former
        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN

        offset = offset.view(B, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()  # b Ca t p p
        offset = self.hs_attention(offset)
        torch.backends.cudnn.enabled = cudnn_enabled

        offset = offset.permute(0, 2, 3, 4, 1).contiguous().view(BT, L - 1, C)
        x_id[:, 1:, :] += offset

        return x_id

class StatisticAttentionBlock(nn.Module):
    """
        HS-Attention block, statistic attention block
        include:
            1. down-channel:
                channel reduction for speed and channel shuffle、
            2. get_moments:
                calculate moments
            3. sta_distribute：
                distribute statistics based on the similarity
            4. up-channel:
                channel recover + residual connection
    """
    def __init__(
            self,
            in_channels,
            adapter_channels,
            moments=None,
            moment_norm=True):
        super(StatisticAttentionBlock, self).__init__()
        self.in_channels = in_channels
        self.adapter_channels = adapter_channels

        self.moments = [1, 2, 4, 5, 6] if moments is None else moments
        self.moment_norm = moment_norm

        self.down_channel = nn.Conv3d(self.in_channels, self.adapter_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.up_channel = nn.Sequential(
            nn.Conv3d(self.adapter_channels, self.in_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(self.in_channels))

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        nn.init.constant_(self.up_channel[-1].weight.data, 0.0)
        nn.init.constant_(self.up_channel[-1].bias.data, 0.0)

    def forward(self, x):
        z = self.down_channel(x)
        y = get_moments(z, moments=self.moments, moment_norm=self.moment_norm)
        z = sta_distribute(z, y)
        z = self.up_channel(z)
        z = z + x
        z = F.relu(z)
        return z

def get_moments(z, moments, moment_norm=True):
    """
    :param z:               (b, c, t, h, w)
    :param moments:         e.g. [1, 2, 3, 4]
    :param moment_norm:     True or False
    :return:
        (b, c, m), m=|moments|
    """
    b, c, t, h, w = z.size()

    mean = F.adaptive_avg_pool3d(z, output_size=1)  # (b, c, 1, 1, 1)
    mean = mean.reshape(b, c, 1)
    moments_set = [mean, ]

    z = z.reshape(b, c, t * h * w)  # (b, c, t*h*w)

    if 2 in moments:
        variance = torch.mean((z - torch.mean(z, dim=-1, keepdim=True)) ** 2, dim=-1, keepdim=True)
        if moment_norm: variance = torch.sqrt(variance)
        moments_set.append(variance)

        for i in moments:
            if i <= 2: continue

            c_moment = torch.mean((z - torch.mean(z, dim=-1, keepdim=True)) ** i, dim=-1, keepdim=True)
            if moment_norm:
                c_moment = c_moment / (variance ** i)
            moments_set.append(c_moment)

    y = torch.cat(moments_set, dim=2)   # (b, c, m)
    return y

def sta_distribute(x, mv, norm=True):
    """
    :param x:   feature map, size: (b, c, t, h, w)
    :param mv:  moment vectors, size: (b, c, m)
    :param norm:    True or False
    :return:
        (b, c, t, h, w)
    """
    b, c, t, h, w = x.size()

    if norm:
        mv = F.normalize(mv, dim=1)  # (b, c, m)

    x = x.reshape(b, c, t * h * w).permute(0, 2, 1)  # (b, t*h*w, c)
    mv = mv.reshape(b, c, -1)

    f = torch.matmul(x, mv)     # (b, t*h*w, m)
    f = F.softmax(f, dim=-1)    # (b, t*h*w, m)

    # (b, t*h*w, c) <- (b, t*h*w, m) * (b, m, c)
    x = torch.matmul(f, mv.permute(0, 2, 1))
    x = x.permute(0, 2, 1).reshape(b, c, t, h, w)

    return x


class OffsetAdapter(nn.Module):

    def __init__(self, in_channels, adapter_channels, kernel_size, T):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv3d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=tuple(x // 2 for x in kernel_size),
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        self.T = T
        self.adapter_channels = adapter_channels
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        T = self.T
        BT, L, C = x.size()
        B = BT // T
        Ca = self.adapter_channels
        H = W = round(math.sqrt(L - 1))
        assert L - 1 == H * W
        x = x[:, 1:, :].view(B, T, -1, C)
        former_id = [0] + [i for i in range(T)][:-1]
        x_former = x[:, former_id]

        offset = x - x_former
        offset = self.fc1(offset)
        offset = offset.view(B, T, H, W, Ca).permute(0, 4, 1, 2, 3).contiguous()  # b Ca t p p

        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        offset = self.conv(offset)
        torch.backends.cudnn.enabled = cudnn_enabled

        offset = offset.permute(0, 2, 3, 4, 1).contiguous().view(BT, L - 1, Ca)
        offset = self.fc2(offset)

        return offset


class T_Adapter(nn.Module):

    def __init__(self, in_channels, adapter_channels):
        super().__init__()
        self.textad_fc1 = nn.Linear(in_channels, adapter_channels)
        self.textad_gelu = nn.GELU()
        self.textad_relu = nn.ReLU()  # 保留原始并行分支的ReLU
        self.textad_fc2 = nn.Linear(adapter_channels, in_channels)

        # 调用自定义初始化方法
        self._init_weights()

    def _init_weights(self):
        """优化初始化策略，适配GELU和ReLU的特性"""
        # 1. 初始化textad_fc1（输入→中间层）
        # GELU分支对权重更敏感，使用较小的初始化范围；ReLU分支适合Kaiming初始化
        # 由于fc1同时服务于两个分支，采用折中策略：GELU的近似增益 × Kaiming初始化
        gelu_gain = 0.8796  # GELU的近似增益（源自tanh近似）
        nn.init.kaiming_uniform_(self.textad_fc1.weight, mode='fan_in', nonlinearity='linear')
        self.textad_fc1.weight.data *= gelu_gain  # 应用GELU的增益调整

        # 2. 初始化textad_fc2（中间层→输出）
        # 输出层需要稳定残差连接，使用Xavier初始化（适合线性变换）
        nn.init.xavier_uniform_(self.textad_fc2.weight, gain=1.0)

        # 3. 偏置初始化保持原始的0值（避免引入额外偏差）
        nn.init.constant_(self.textad_fc1.bias, 0.)
        nn.init.constant_(self.textad_fc2.bias, 0.)

    def forward(self, x):
        # 保持原始并行分支逻辑不变
        x1 = self.textad_fc1(x)
        x1 = self.textad_gelu(x1)
        x1 = self.textad_fc2(x1)

        x2 = self.textad_fc1(x)
        x2 = self.textad_relu(x2)
        x2 = self.textad_fc2(x2)

        x = x + x1 + x2  # 合并残差（简化原始的两次加法）
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        # pdb.set_trace()
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 adapter_width: int,
                 adapter_kernel_size: Tuple[int, int, int],
                 adapter_earlier_attn: bool,
                 adapter_later_attn: bool,
                 adapter_earlier_mlp: bool,
                 adapter_later_mlp: bool,
                 num_frames: int,
                 enable_hs: bool = True,
                 attn_mask: torch.Tensor = None
                 ) -> None:
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.T = num_frames
        self.attn_mask = attn_mask

        hs_adapter_earlier_class = functools.partial(
            HS_Adapter_Earlier,
            in_channels=d_model,
            adapter_channels=adapter_width,
            kernel_size=3,
            T=self.T
        )

        #lm
        hs_adapter_later_class = functools.partial(
            HS_Adapter_Later,
            in_channels=d_model,
            adapter_channels=adapter_width,
            kernel_size=(1,3,3),
            T=self.T
        )

        text_adapter_class = functools.partial(
            T_Adapter,
            in_channels=d_model,
            adapter_channels=adapter_width
        )

        if num_frames == 0:
            self.adapter_earlier_attn = text_adapter_class() if adapter_earlier_attn else None
            self.adapter_later_attn = None
            self.adapter_earlier_mlp = text_adapter_class() if adapter_earlier_mlp else None
            self.adapter_later_mlp = None
        else:
            self.adapter_earlier_attn = hs_adapter_earlier_class() if adapter_earlier_attn else None
            self.adapter_later_attn = hs_adapter_later_class() if adapter_later_attn else None
            self.adapter_earlier_mlp = hs_adapter_earlier_class() if adapter_earlier_mlp else None
            self.adapter_later_mlp = hs_adapter_later_class() if adapter_later_mlp else None
            # self.hs_adapter_later = hs_adapter_later_class() if enable_hs else None

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        B, L, C = x.size()
        x = x.permute(1, 0, 2)
        x = self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

        return x.permute(1, 0, 2)

    def cross_attention(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:

        B, L, C = x.size()

        x = x.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        x = self.attn(x, k, k, need_weights=False, attn_mask=None)[0]

        return x.permute(1, 0, 2)

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:

        if self.adapter_earlier_attn is not None:
            x = self.adapter_earlier_attn(x)
        if self.adapter_later_attn is not None:
            x = self.adapter_later_attn(x)


        x = x + self.attention(self.ln_1(x))


        x = x + self.mlp(self.ln_2(x))
        if self.adapter_earlier_mlp is not None:
            x = self.adapter_earlier_mlp(x)
        if self.adapter_later_mlp is not None:
            x = self.adapter_later_mlp(x)


        return x

    def forward_cross(self,
                      x: torch.Tensor,
                      k: torch.Tensor
                      ) -> torch.Tensor:

        if self.adapter_earlier_attn is not None:
            x = self.adapter_earlier_attn(x)
        if self.adapter_later_attn is not None:
            x = self.adapter_later_attn(x)
        x = x + self.cross_attention(self.ln_1(x), self.ln_1(k))
        if self.adapter_earlier_mlp is not None:
            x = self.adapter_earlier_mlp(x)
        if self.adapter_later_mlp is not None:
            x = self.adapter_later_mlp(x)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 adapter_width: int,
                 adapter_layers: int,
                 adapter_kernel_size: Tuple[int, int, int],
                 adapter_earlier_attn: bool,
                 adapter_later_attn: bool,
                 adapter_earlier_mlp: bool,
                 adapter_later_mlp: bool,
                 num_frames: int,
                 hs_indices: List[int] = None,
                 attn_mask: torch.Tensor = None
                 ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.hs_indices = hs_indices or []  # 默认空列表（不启用）
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                d_model=width,
                n_head=heads,
                adapter_width=adapter_width,
                adapter_kernel_size=adapter_kernel_size,
                adapter_earlier_attn=adapter_earlier_attn and i >= layers - adapter_layers,
                adapter_later_attn=adapter_later_attn and i in self.hs_indices,
                adapter_earlier_mlp=adapter_earlier_mlp and i >= layers - adapter_layers,
                adapter_later_mlp=adapter_later_mlp and i in self.hs_indices,
                num_frames=num_frames,
                attn_mask=attn_mask,
                # 检查当前层索引是否在指定列表中
                # enable_hs=(i in self.hs_indices)
            )
            for i in range(layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, block in enumerate(self.resblocks):
            if i < 12:
                x = block(x)
        return x



class VisionTransformer(nn.Module):
    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 output_dim: int,
                 adapter_width: int,
                 adapter_layers: int,
                 adapter_kernel_size: Tuple[int, int, int],
                 adapter_earlier_attn: bool,
                 adapter_later_attn: bool,
                 adapter_earlier_mlp: bool,
                 adapter_later_mlp: bool,
                 num_classes: int,
                 num_frames: int,
                 class_fc: bool = True,
                 hs_indices: List[int] = None  # 传递层索引列表
                 ):
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width,
                               kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(
                (input_resolution // patch_size) ** 2 + 1, width
            )
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads,
                                       adapter_width, adapter_layers, adapter_kernel_size,
                                       adapter_earlier_attn, adapter_later_attn, adapter_earlier_mlp, adapter_later_mlp, num_frames, hs_indices)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.t_adapter = Adapter1(
            in_channels=output_dim,  # 512
            adapter_channels=output_dim // 2,  # 256
            kernel_size=3,
            T=num_frames
        )

        self.adapter_dropout = nn.Dropout(0.5)

        self.adapter_fc = nn.Linear(width, num_classes) if class_fc else None
        if self.adapter_fc is not None:
            nn.init.normal_(self.adapter_fc.weight, std=0.02)
            nn.init.constant_(self.adapter_fc.bias, 0.)

    def forward(self, x: torch.Tensor):
        B, T = x.size(0), x.size(2)  # b c t h w
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)  # bt c h w
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        spatial_size = tuple(x.size()[2:])  # p p
        x = x.flatten(-2).permute(0, 2, 1)  # bt pp c
        x = torch.cat([
            self.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1), x
        ], dim=1)  # [*, grid ** 2 + 1, width]  # bt cls+pp c
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = self.transformer(x)

        if self.adapter_fc is not None:
            x2 = x.contiguous().view(B, T, spatial_size[0] * spatial_size[1] + 1, x.size(-1))  # b t pp+cls c
            x2 = x2[:, :, 0, :].mean(dim=1)  # b t cls c -> b c  clstoken

            x2 = self.ln_post(x2)
            x2 = self.adapter_dropout(x2)
            x2 = self.adapter_fc(x2)
        else:
            x2 = None  # (b, num_classes)

        x1 = self.ln_post(x)

        x1 = x1[:, 0, :] @ self.proj
        x1 = self.t_adapter(x1)#x_id  #lm
        x = x1.view(B, T, -1).mean(dim=1, keepdim=False)

        return x, x1.view(B, T, -1), x2

        #return x, x1, x2


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 vision_adapter_width: int,
                 vision_adapter_layers: int,
                 vision_adapter_kernel_size: Tuple[int, int, int],
                 vision_adapter_earlier_attn: bool,
                 vision_adapter_later_attn: bool,
                 vision_adapter_earlier_mlp: bool,
                 vision_adapter_later_mlp: bool,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 text_adapter_width: int,
                 text_adapter_layers: int,
                 text_adapter_kernel_size: Tuple[int, int, int],
                 text_adapter_earlier_attn: bool,
                 text_adapter_later_attn: bool,
                 text_adapter_earlier_mlp: bool,
                 text_adapter_later_mlp: bool,
                 num_classes: int,
                 num_frames: int,
                 mlm_head_len: int,
                 hs_indices: List[int],
                 ):
        super().__init__()

        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            adapter_width=vision_adapter_width,
            adapter_layers=vision_adapter_layers,
            adapter_kernel_size=vision_adapter_kernel_size,
            adapter_earlier_attn=vision_adapter_earlier_attn,
            adapter_later_attn=vision_adapter_later_attn,
            adapter_earlier_mlp=vision_adapter_earlier_mlp,
            adapter_later_mlp=vision_adapter_later_mlp,
            output_dim=embed_dim,
            num_classes=num_classes,
            class_fc=True,
            num_frames=num_frames,
            hs_indices=hs_indices,
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            adapter_width=text_adapter_width,
            adapter_layers=text_adapter_layers,
            adapter_kernel_size=text_adapter_kernel_size,
            adapter_earlier_attn=text_adapter_earlier_attn,
            adapter_later_attn=text_adapter_later_attn,
            adapter_earlier_mlp=text_adapter_earlier_mlp,
            adapter_later_mlp=text_adapter_later_mlp,
            num_frames=0,
        )

        self.vocab_size = vocab_size
        self.mlm_head_len = mlm_head_len
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.config = BertConfig(
            vocab_size=mlm_head_len,
            hidden_size=transformer_width
        )

        self.initialize_parameters()
        self.mlm_adapter = MLMHead(self.config)
        self.mlm_adapter.apply(self.init_mlm_weights)
    def init_mlm_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):  # 16 77

        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model] 16 77 512

        x = x + self.positional_embedding.type(self.dtype)
        x1 = self.transformer(x)
        x = self.ln_final(x1).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]  # 16 512
        x = x @ self.text_projection  # 16 512

        return x, x1.type(self.dtype)  # 16 512    16 77 512

    #                      16 77 512,   16 77,     16 8 512
    def compute_mlm(self, text_feats, mlm_labels, visual_feats):  # DEBUG
        text_feats = self.transformer.resblocks[12].forward_cross(text_feats, visual_feats)
        text_feats = self.ln_final(text_feats)
        mlm_logits = self.mlm_adapter(text_feats)  # 16 77 520

        mlm_loss = F.cross_entropy(
            mlm_logits.view(-1, self.mlm_head_len),  # 520
            mlm_labels.view(-1),
            ignore_index=-100,
        )
        ret = {
            "mlm_loss": mlm_loss,
            # "mlm_logits": mlm_logits,
            # "mlm_labels": mlm_labels
        }

        return ret

    #          16 3 8 224 224 , 16 77
    def forward(self, image, text):
        # 16 512          16 8 512        16 400
        image_features, image_mlm_inputs, image_fc = self.encode_image(image)
        # 16 512         16 77 512
        text_features, text_mlm_inputs = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)  # 16 512
        text_features = text_features / text_features.norm(dim=1, keepdim=True)  # 16 512

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()  # 16 16
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text, image_features, text_mlm_inputs, image_mlm_inputs, image_fc


def copy_weights(source_module, target_module):
    source_params = dict(source_module.named_parameters())
    target_params = dict(target_module.named_parameters())

    for target_name, target_param in target_params.items():
        # pdb.set_trace()
        if target_name in source_params:
            source_param = source_params[target_name]
            if source_param.data.shape == target_param.data.shape:
                target_param.data.copy_(source_param.data)
            else:
                print(f"Warning: Shapes mismatch for parameters {target_name}. Skipping copying.")
        else:
            print(f"Warning: Parameter {target_name} not found in source module. Skipping copying.")


def clip_vit_base_patch16_adapter24x384(**kwargs):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        adapter_width=384,
        adapter_layers=12,
        adapter_kernel_size=(3, 1, 1),
        adapter_earlier_attn=True,
        adapter_later_attn=True,
        adapter_earlier_mlp=True,
        adapter_later_mlp=True,
        **kwargs,
    )
    assert CLIP_VIT_B16_PATH is not None, \
        'Please set CLIP_VIT_B16_PATH in configs.py.'
    checkpoint = torch.jit.load(CLIP_VIT_B16_PATH, map_location='cpu')
    print(model.load_state_dict(checkpoint.visual.state_dict(), strict=False))
    return model


def clip_vit_base_patch16_adapter12x384(**kwargs):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        adapter_width=384,
        adapter_layers=12,
        adapter_kernel_size=(3, 1, 1),
        adapter_earlier_attn=True,
        adapter_later_attn=True,
        adapter_earlier_mlp=True,
        adapter_later_mlp=True,
        **kwargs,
    )
    assert CLIP_VIT_B16_PATH is not None, \
        'Please set CLIP_VIT_B16_PATH in configs.py'
    checkpoint = torch.jit.load(CLIP_VIT_B16_PATH, map_location='cpu')
    print(model.load_state_dict(checkpoint.visual.state_dict(), strict=False))
    return model


def clip_vit_base_patch32_adapter12x384(**kwargs):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=32,
        width=768,
        layers=12,
        heads=12,
        adapter_width=384,
        adapter_layers=12,
        adapter_kernel_size=(3, 1, 1),
        adapter_earlier_attn=True,
        adapter_later_attn=True,
        adapter_earlier_mlp=True,
        adapter_later_mlp=True,
        **kwargs,
    )
    assert CLIP_VIT_B32_PATH is not None, \
        'Please set CLIP_VIT_B32_PATH in configs.py'
    checkpoint = torch.jit.load(CLIP_VIT_B32_PATH, map_location='cpu')
    print(model.load_state_dict(checkpoint.visual.state_dict(), strict=False))
    model.load_state_dict(checkpoint.visual.state_dict(), strict=False)
    return model


def clip_vit_base_patch32_adapter24x384(**kwargs):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=32,
        width=768,
        layers=12,
        heads=12,
        adapter_width=384,
        adapter_layers=12,
        adapter_kernel_size=(3, 1, 1),
        adapter_earlier_attn=True,
        adapter_later_attn=True,
        adapter_earlier_mlp=True,
        adapter_later_mlp=True,
        **kwargs,
    )
    assert CLIP_VIT_B32_PATH is not None, \
        'Please set CLIP_VIT_B32_PATH in configs.py'
    checkpoint = torch.jit.load(CLIP_VIT_B32_PATH, map_location='cpu')
    model.load_state_dict(checkpoint.visual.state_dict(), strict=False)
    return model



def clip_vit_base_patch32_multimodal_adapter12x384(**kwargs):
    checkpoint = torch.jit.load(CLIP_VIT_B32_PATH, map_location='cpu')
    model = CLIP(
        embed_dim=512,
        image_resolution=224,
        vision_patch_size=32,
        vision_width=768,
        vision_layers=12,
        vision_adapter_width=384,
        vision_adapter_layers=12,
        vision_adapter_kernel_size=(3, 1, 1),
        vision_adapter_earier_attn=True,
        vision_adapter_later_attn=False,
        vision_adapter_earier_mlp=False,
        vision_adapter_later_mlp=True,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=13,
        text_adapter_width=384,
        text_adapter_layers=1,
        text_adapter_kernel_size=(3, 1, 1),
        text_adapter_earlier_attn=False,
        text_adapter_later_attn=False,
        text_adapter_earlier_mlp=True,
        text_adapter_later_mlp=False,
        hs_indices=[4, 8, 12],
        **kwargs,
    )
    # pdb.set_trace()
    assert CLIP_VIT_B32_PATH is not None, \
        'Please set CLIP_VIT_B32_PATH in configs.py'
    # pdb.set_trace()
    # convert_weights(model)

    # print(model.load_state_dict(checkpoint.visual.state_dict(), strict=False))
    model.load_state_dict(checkpoint.state_dict(), strict=False)

    copy_weights(model.transformer.resblocks[11], model.transformer.resblocks[12])
    convert_weights(model)
    return model


def clip_vit_base_patch16_multimodal_adapter12x384(**kwargs):
    checkpoint = torch.jit.load(CLIP_VIT_B16_PATH, map_location='cpu')
    model = CLIP(
        embed_dim=512,
        image_resolution=224,
        vision_patch_size=16,
        vision_width=768,
        vision_layers=12,
        vision_adapter_width=384,
        vision_adapter_layers=12,
        vision_adapter_kernel_size=(3, 1, 1),
        vision_adapter_earier_attn=True,
        vision_adapter_later_attn=True,
        vision_adapter_earier_mlp=False,
        vision_adapter_later_mlp=False,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=13,
        text_adapter_width=384,
        text_adapter_layers=2,
        text_adapter_kernel_size=(3, 1, 1),
        text_adapter_earlier_attn=False,
        text_adapter_later_attn=False,
        text_adapter_earlier_mlp=True,
        text_adapter_later_mlp=True,
        hs_indices=[4, 8, 12],
        **kwargs,
    )
    assert CLIP_VIT_B16_PATH is not None, \
        'Please set CLIP_VIT_B16_PATH in configs.py'
    # pdb.set_trace()
    # convert_weights(model)

    # print(model.load_state_dict(checkpoint.visual.state_dict(), strict=False))
    model.load_state_dict(checkpoint.state_dict(), strict=False)

    copy_weights(model.transformer.resblocks[11], model.transformer.resblocks[12])
    convert_weights(model)
    return model
