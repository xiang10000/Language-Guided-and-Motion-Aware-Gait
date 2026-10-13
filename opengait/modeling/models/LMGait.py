# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from ..base_model import BaseModel
from torch.nn import functional as F
from kornia import morphology as morph
import random
from functools import partial

# import text utils
from ..text_utils import text_tokenizer
from typing import Tuple
from collections import OrderedDict
import functools
import math
import numpy as np
from typing import Tuple, Union
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform, BertConfig
from ..modules import SetBlockWrapper, SeparateFCs, SeparateBNNecks, PackSequenceWrapper, HorizontalPoolingPyramid

from ..text_configs import (
    CLIP_VIT_B16_PATH,
    CLIP_VIT_B32_PATH,
    CLIP_VIT_L14_PATH,
    DWCONV3D_DISABLE_CUDNN,
    )

# import GaitBase & DINOv2_small
from .BigGait_utils.BigGait_GaitBase import Baseline
from .BigGait_utils.DINOv2 import vit_small
from .BigGait_utils.save_img import save_image, pca_image

# ######################################## BigGait ###########################################

class infoDistillation(nn.Module):
    def __init__(self, source_dim, target_dim, p, softmax, Relu, Up=True):
        super(infoDistillation, self).__init__()
        self.dropout = nn.Dropout(p=p)
        self.bn_s = nn.BatchNorm1d(source_dim, affine=False)
        self.bn_t = nn.BatchNorm1d(target_dim, affine=False)
        if Relu:
            self.down_sampling = nn.Sequential(
                nn.Linear(source_dim, source_dim//2),
                nn.BatchNorm1d(source_dim//2, affine=False),
                nn.GELU(),
                nn.Linear(source_dim//2, target_dim),
                )
            if Up:
                self.up_sampling = nn.Sequential(
                    nn.Linear(target_dim, source_dim//2),
                    nn.BatchNorm1d(source_dim//2, affine=False),
                    nn.GELU(),
                    nn.Linear(source_dim//2, source_dim),
                    )
        else:
            self.down_sampling = nn.Linear(source_dim, target_dim)
            if Up:
                self.up_sampling = nn.Linear(target_dim, source_dim)
        self.softmax = softmax
        self.mse = nn.MSELoss()
        self.Up = Up

    def forward(self, x):
        # [n, c]
        d_x = self.down_sampling(self.bn_s(self.dropout(x)))
        if self.softmax:
            d_x = F.softmax(d_x, dim=1)
            if self.Up:
                u_x = self.up_sampling(d_x)
                return d_x, torch.mean(self.mse(u_x, x))
            else:
                return d_x, None
        else:
            if self.Up:
                u_x = self.up_sampling(d_x)
                return torch.sigmoid(self.bn_t(d_x)), torch.mean(self.mse(u_x, x))
            else:
                return torch.sigmoid(self.bn_t(d_x)), None


class TextAdapter(nn.Module):

    def __init__(self, in_channels, adapter_channels):
        super().__init__()
        self.textad_fc1 = nn.Linear(in_channels, adapter_channels)
        self.textad_gelu = nn.GELU()
        self.textad_fc2 = nn.Linear(adapter_channels, in_channels)
        nn.init.constant_(self.textad_fc1.bias, 0.)
        nn.init.constant_(self.textad_fc2.bias, 0.)

    def forward(self, x):
        # pdb.set_trace()
        x1 = self.textad_fc1(x)
        x1 = self.textad_gelu(x1)
        x1 = self.textad_fc2(x1)
        x = x + x1
        return x

class OffsetAdapter(nn.Module):

    def __init__(self, in_channels, adapter_channels, kernel_size,T,B):
        # def __init__(self, in_channels, adapter_channels, kernel_size,T):
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
        self.B = B
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x, B=None):
        # T = self.T
        B = self.B if B is None else B
        BT, L, C = x.size()
        # B = BT // T
        T = BT // B
        Ca = self.conv.in_channels
        Ca= 64 # ji de shan chu
        # H = W = round(math.sqrt(L - 1))
        # assert L - 1 == H * W
        W = int(math.sqrt(L / 2))
        H = 2 * W
        assert H * W == L, f"Shape mismatch: H*W={H*W} != L={L}"
        x_id = x
        # x = x[:, 1:, :].view(B,T,-1,C)
        x = x.view(B,T,-1,C)
        # offset = x
        former_id = [0] + [i for i in range(T)][:-1]
        x_former = x[:,former_id]
        # # pdb.set_trace()
        offset = x - x_former
        # offset = self.fc1(offset)
        offset = offset.view(B, T, H, W, Ca).permute(0, 4, 1, 2, 3).contiguous()
        offset = self.conv(offset)

        # offset = offset.permute(0, 2, 3, 4, 1).contiguous().view(BT, L - 1, Ca)
        offset = offset.permute(0, 2, 3, 4, 1).contiguous().view(BT, L, Ca)
        # offset = self.fc2(offset)
        # offset = offset + x_id
        return offset

# class Adapter(nn.Module):

#     def __init__(self, in_channels, adapter_channels, kernel_size,T):
#         super().__init__()
#         self.fc1 = nn.Linear(in_channels, adapter_channels)
#         self.conv = nn.Conv3d(
#             adapter_channels, adapter_channels,
#             kernel_size=kernel_size,
#             stride=(1, 1, 1),
#             padding=tuple(x // 2 for x in kernel_size),
#             groups=adapter_channels,
#         )
#         self.fc2 = nn.Linear(adapter_channels, in_channels)
#         self.T = T
#         self.offset_adapter = OffsetAdapter(in_channels,adapter_channels,(1,3,3),T)
#         nn.init.constant_(self.conv.weight, 0.)
#         nn.init.constant_(self.conv.bias, 0.)
#         nn.init.constant_(self.fc1.bias, 0.)
#         nn.init.constant_(self.fc2.bias, 0.)

#     def forward(self, x):
#         offset = self.offset_adapter(x)
#         T = self.T
#         BT, L, C = x.size()
#         B = BT // T
#         Ca = self.conv.in_channels
#         H = W = round(math.sqrt(L - 1))
        
#         assert L - 1 == H * W
#         x_id = x
#         x = x[:, 1:, :]
#         x = self.fc1(x)
#         x = x.view(B, T, H, W, Ca).permute(0, 4, 1, 2, 3).contiguous()

#         x = self.conv(x)

#         x = x.permute(0, 2, 3, 4, 1).contiguous().view(BT, L - 1, Ca)
#         x = self.fc2(x) + offset
#         x_id[:, 1:, :] += x
#         return x_id


# class Adapter1(nn.Module):

#     def __init__(self, in_channels, adapter_channels, kernel_size,T):
#         super().__init__()
#         self.fc1 = nn.Linear(in_channels, adapter_channels)
#         self.conv = nn.Conv1d(
#             adapter_channels, adapter_channels,
#             kernel_size=kernel_size,
#             stride=1,
#             padding=kernel_size // 2,
#             groups=adapter_channels,
#         )
#         self.fc2 = nn.Linear(adapter_channels, in_channels)
#         self.T = T
#         nn.init.constant_(self.conv.weight, 0.)
#         nn.init.constant_(self.conv.bias, 0.)
#         nn.init.constant_(self.fc1.bias, 0.)
#         nn.init.constant_(self.fc2.bias, 0.)

#     def forward(self, x):
#         T = self.T
#         BT, C = x.size()
#         B = BT // T
#         Ca = self.conv.in_channels
#         x_id = x
#         x = self.fc1(x)
#         x = x.view(B, T, Ca).permute(0, 2, 1).contiguous().view(B,Ca,T)  #

#         cudnn_enabled = torch.backends.cudnn.enabled
#         torch.backends.cudnn.enabled = cudnn_enabled
#         x = self.conv(x)
#         torch.backends.cudnn.enabled = cudnn_enabled

#         x = x.permute(0, 2, 1).contiguous().view(-1,Ca)
#         x = self.fc2(x)
#         x_id = x + x_id
#         # pdb.set_trace()
#         return x_id

class Adapter2(nn.Module):
    def __init__(self, in_channels, adapter_channels, kernel_size,T,B):
        # def __init__(self, in_channels, adapter_channels, kernel_size,T):
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
        self.B = B
        # self.offset_adapter = OffsetAdapter(in_channels,adapter_channels,(1,3,3),T)
        self.offset_adapter = OffsetAdapter(in_channels,adapter_channels,(1,3,3),T,B)
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x, B=None):
        B = self.B if B is None else B
        offset = self.offset_adapter(x, B)
        # T = self.T
        BT,L, C = x.size()
        # B = BT // T
        T = BT // B
        Ca = self.conv.in_channels
        Ca = 64
        x_id = x
        # x = self.fc1(x)
        # Ca = 1536 # ji de zhu shi
        x = x.view(B, T,L, Ca).permute(0, 2,3, 1).contiguous().view(B*L,Ca,T)  #

        x = self.conv(x)

        x = x.permute(0, 2, 1).contiguous().view(B,L,T,Ca)
        x = x.permute(0, 2, 1,3).contiguous().view(BT,L,Ca)
        # x = self.fc2(x)
        x_id = x_id + offset
        x_id = x + x_id

        
        return x_id

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
                 adapter_pre_attn: bool,
                 adapter_pre_mlp: bool,
                 num_frames: int,
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

        # adapter_class = functools.partial(
        #     Adapter2,
        #     in_channels=d_model,
        #     adapter_channels=adapter_width,
        #     kernel_size=3,
        #     T=self.T
        # )
        # offset_adapter_class = functools.partial(
        #     OffsetAdapter,
        #     in_channels=d_model,
        #     adapter_channels=adapter_width,
        #     kernel_size=(1,3,3),
        #     T=self.T
        # )
        text_adapter_class = functools.partial(
            TextAdapter,
            in_channels=d_model,
            adapter_channels=adapter_width
        )
        if num_frames==0:
            self.adapter_pre_attn = \
                text_adapter_class() if adapter_pre_attn else None
            self.adapter_pre_attn_off = None
            self.adapter_pre_mlp = \
                text_adapter_class() if adapter_pre_mlp else None
            self.adapter_pre_mlp_off = None
        # else:    
        #     self.adapter_pre_attn = \
        #         adapter_class() if adapter_pre_attn else None
        #     self.adapter_pre_attn_off = None
        #     self.adapter_pre_mlp = \
        #         adapter_class() if adapter_pre_mlp else None
        #     self.adapter_pre_mlp_off = None

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        B, L, C = x.size()
        # pdb.set_trace()
        # H = self.attn.num_heads

        # qkv = F.linear(x, weight=self.attn.in_proj_weight, bias=self.attn.in_proj_bias)
        # qkv = qkv.view(B, L, H * 3, -1).permute(0, 2, 1, 3)
        # q, k, v = qkv.split([H, H, H], dim=1)
        # out = F.scaled_dot_product_attention(q, k, v)
        # out = out.permute(0, 2, 1, 3).flatten(-2)
        # out = self.attn.out_proj(out)

        # return out
        x = x.permute(1,0,2)
        # print(x.shape)
        # pdb.set_trace()
        x = self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

        return x.permute(1,0,2)

    def cross_attention(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        
        B, L, C = x.size()
        # pdb.set_trace()
        # H = self.attn.num_heads

        # qkv = F.linear(x, weight=self.attn.in_proj_weight, bias=self.attn.in_proj_bias)
        # qkv = qkv.view(B, L, H * 3, -1).permute(0, 2, 1, 3)
        # q, k, v = qkv.split([H, H, H], dim=1)
        # out = F.scaled_dot_product_attention(q, k, v)
        # out = out.permute(0, 2, 1, 3).flatten(-2)
        # out = self.attn.out_proj(out)

        # return out
        x = x.permute(1,0,2)
        k = k.permute(1,0,2)
        # print(x.shape)
        # pdb.set_trace()
        x = self.attn(x, k, k, need_weights=False, attn_mask=None)[0]

        return x.permute(1,0,2)
        

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        
        # pdb.set_trace()
        if self.adapter_pre_attn_off is not None:
            x = self.adapter_pre_attn_off(x)
        if self.adapter_pre_attn is not None:
            x = self.adapter_pre_attn(x)
        x = x + self.attention(self.ln_1(x))
        if self.adapter_pre_mlp_off is not None:
            x = self.adapter_pre_mlp_off(x)
        if self.adapter_pre_mlp is not None:
            x = self.adapter_pre_mlp(x)
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward_cross(self,
                x: torch.Tensor,
                k: torch.Tensor
                ) -> torch.Tensor:
        
        # pdb.set_trace()
        if self.adapter_pre_attn is not None:
            x = self.adapter_pre_attn(x)
        x = x + self.cross_attention(self.ln_1(x),self.ln_1(k))
        if self.adapter_pre_mlp is not None:
            x = self.adapter_pre_mlp(x)
        x = x + self.mlp(self.ln_2(x))
        return x

class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        # self.ln_final = LayerNorm(config.hidden_size)
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        # x = self.ln_final(x)
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x

class Transformer(nn.Module):
    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 adapter_width: int,
                 adapter_layers: int,
                 adapter_kernel_size: Tuple[int, int, int],
                 adapter_pre_attn: bool,
                 adapter_pre_mlp: bool,
                 num_frames: int,
                 attn_mask: torch.Tensor = None,
                 ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                d_model=width,
                n_head=heads,
                adapter_width=adapter_width,
                adapter_kernel_size=adapter_kernel_size,
                adapter_pre_attn=adapter_pre_attn and i >= layers - adapter_layers,
                adapter_pre_mlp=adapter_pre_mlp and i >= layers - adapter_layers,
                num_frames=num_frames,
                attn_mask=attn_mask
            )
            for i in range(layers)
        ])
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, block in enumerate(self.resblocks):
            x = block(x)
        return x
# class VisionTransformer(nn.Module):
#     def __init__(self,
#                  input_resolution: int,
#                  patch_size: int,
#                  width: int,
#                  layers: int,
#                  heads: int,
#                  output_dim: int,
#                  adapter_width: int,
#                  adapter_layers: int,
#                  adapter_kernel_size: Tuple[int, int, int],
#                  adapter_pre_attn: bool,
#                  adapter_pre_mlp: bool,
#                  num_classes: int,
#                  num_frames:int,
#                  class_fc: bool = True
#                  ):
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=width,
#             kernel_size=patch_size, stride=patch_size, bias=False)

#         scale = width ** -0.5
#         self.class_embedding = nn.Parameter(scale * torch.randn(width))
#         self.positional_embedding = nn.Parameter(
#             scale * torch.randn(
#                 (input_resolution // patch_size) ** 2 + 1, width
#             )
#         )
#         self.ln_pre = LayerNorm(width)

#         self.transformer = Transformer(width, layers, heads,
#             adapter_width, adapter_layers, adapter_kernel_size,
#             adapter_pre_attn, adapter_pre_mlp,num_frames)

#         self.ln_post = LayerNorm(width)
#         self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
#         self.t_adapter = Adapter1(
#             in_channels=output_dim,
#             adapter_channels=output_dim//2,
#             kernel_size=3,
#             T=num_frames
#         )
#         # for n, p in self.named_parameters():
#         #   if 'adapter' not in n:
#         #     p.requires_grad_(False)
#         #     p.data = p.data.half()
        
#         self.adapter_dropout = nn.Dropout(0.5)
        
#         self.adapter_fc = \
#             nn.Linear(width, num_classes) if class_fc else None 
#         if self.adapter_fc is not None:
#             nn.init.normal_(self.adapter_fc.weight, std=0.02)
#             nn.init.constant_(self.adapter_fc.bias, 0.)


#     def forward(self, x: torch.Tensor):
#         B, T = x.size(0), x.size(2)
#         # pdb.set_trace()
#         x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)
#         x = self.conv1(x)  # shape = [*, width, grid, grid]
#         spatial_size = tuple(x.size()[2:])
#         x = x.flatten(-2).permute(0, 2, 1)
#         x = torch.cat([
#             self.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1), x
#             ], dim=1)  # [*, grid ** 2 + 1, width]
#         # pdb.set_trace()
#         x = x + self.positional_embedding.to(x.dtype)
#         x = self.ln_pre(x)

#         # x = x.view(B, T, x.size(1), x.size(2)).flatten(0, 1) # BT, L, D
        
#         x, extra_x = self.transformer(x)
#         # pdb.set_trace()
             

#         if self.adapter_fc is not None:
#             x2 = x.contiguous().view(B, T, spatial_size[0] * spatial_size[1] + 1, x.size(-1))
#             x2 = x2[:, :, 0, :].mean(dim=1)

#             x2 = self.ln_post(x2)
#             x2 = self.adapter_dropout(x2)
#             x2 = self.adapter_fc(x2)
#         else:
#             x2= None
        
#         x1 = self.ln_post(x)
        
#         x1 = x1[:,0,:] @ self.proj
#         x1 = self.t_adapter(x1) 
#         x = x1.view(B,T,-1).mean(dim=1, keepdim=False)
            

#         return x,x1.view(B,T,-1),x2


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
                 vision_adapter_pre_attn: bool,
                 vision_adapter_pre_mlp: bool,           
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 text_adapter_width: int,
                 text_adapter_layers: int,
                 text_adapter_kernel_size: Tuple[int, int, int],
                 text_adapter_pre_attn: bool,
                 text_adapter_pre_mlp: bool,  
                 num_classes: int,
                 num_frames: int,
                 mlm_head_len: int
                 ):
        super().__init__()

        self.context_length = context_length

        vision_heads = vision_width // 64
        # self.visual = VisionTransformer(
        #             input_resolution=image_resolution,
        #             patch_size=vision_patch_size,
        #             width=vision_width,
        #             layers=vision_layers,
        #             heads=vision_heads,
        #             adapter_width=vision_adapter_width,
        #             adapter_layers=vision_adapter_layers,
        #             adapter_kernel_size=vision_adapter_kernel_size,
        #             adapter_pre_attn=vision_adapter_pre_attn,
        #             adapter_pre_mlp=vision_adapter_pre_mlp,
        #             output_dim=embed_dim,
        #             num_classes=num_classes,
        #             class_fc=True,
        #             num_frames=num_frames
        # )
        
        
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            adapter_width=text_adapter_width, 
            adapter_layers=text_adapter_layers, 
            adapter_kernel_size=text_adapter_kernel_size,
            adapter_pre_attn=text_adapter_pre_attn,
            adapter_pre_mlp=text_adapter_pre_mlp,
            num_frames=0
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

        
        self.adapter_fc = nn.Linear(transformer_width, 384)

        # for n, p in self.named_parameters():
        #   if 'adapter' not in n:
        #     p.requires_grad_(False)
        #     p.data = p.data.half()
    def init_mlm_weights(self,module):
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
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    @property
    def dtype(self):
        return nn.Conv2d(in_channels=3, out_channels=768,
            kernel_size=16, stride=16, bias=False).weight.dtype
    
    def cross_attention(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        
        B, L, C = x.size()
        # pdb.set_trace()
        # H = self.attn.num_heads

        # qkv = F.linear(x, weight=self.attn.in_proj_weight, bias=self.attn.in_proj_bias)
        # qkv = qkv.view(B, L, H * 3, -1).permute(0, 2, 1, 3)
        # q, k, v = qkv.split([H, H, H], dim=1)
        # out = F.scaled_dot_product_attention(q, k, v)
        # out = out.permute(0, 2, 1, 3).flatten(-2)
        # out = self.attn.out_proj(out)

        # return out
        x = x.permute(1,0,2)
        k = k.permute(1,0,2)
        # print(x.shape)
        # pdb.set_trace()
        x = self.adapter_attn(x, k, k, need_weights=False, attn_mask=None)[0]

        return x.permute(1,0,2)


    # def encode_text(self, text):
        
    #     x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

    #     x = x + self.positional_embedding.type(self.dtype)
    #     # x = x.permute(1, 0, 2)  # NLD -> LND           
    #     x1, extra_xx = self.transformer(x)  # ResidualAttentionBlock的数量等于self.layers， 每个block都有textAdapter
    #     # # x = x.permute(1, 0, 2)  # LND -> NLD
    #     # x = self.ln_final(x1).type(self.dtype)
    #     # # pdb.set_trace()
    #     # # x.shape = [batch_size, n_ctx, transformer.width]
    #     # # take features from the eot embedding (eot_token is the highest number in each sequence)
    #     # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
    #     # x = x @ self.text_projection
    #     extra_x = []
    #     for xx in extra_xx:
    #         temp = self.ln_final(xx).type(self.dtype)
    #         temp = temp[torch.arange(temp.shape[0]), text.argmax(dim=-1)]
    #         temp = temp @ self.text_projection
    #         tmp  = self.adapter_fc(temp)
    #         extra_x.append(tmp)

    #     # return x,x1.type(self.dtype)
    #     return extra_x
    def encode_text(self, text):

        # print(self.token_embedding.weight.device)

        x = self.token_embedding(text.to(self.token_embedding.weight.device)).type(self.dtype) 
        # x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        # x = x.permute(1, 0, 2)  # NLD -> LND
        x1 = self.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x1).type(self.dtype)
        # pdb.set_trace()
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        x = x @ self.text_projection

        return x,x1.type(self.dtype)

    def get_text_feature(self, dataset_partition, device):
        classes, num_text_aug, text_dict,classes_dict = text_tokenizer(dataset_partition)
        with torch.no_grad():
            text_inputs = classes.to(device)
        
        return text_inputs
    
    def forward(self, text):
        # embed_1, logits = self.encode_image(image)
        text_features,text_mlm_inputs = self.encode_text(text)

        # normalized features
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # # cosine similarity as logits
        # logit_scale = self.logit_scale.exp()
        # logits_per_image = logit_scale * image_features @ text_features.t()  # image to text similar
        # logits_per_text = logits_per_image.t()  # text to image similar

        # shape = [global_batch_size, global_batch_size]
        # return logits_per_image, logits_per_text,image_features,text_mlm_inputs,image_mlm_inputs,image_fc
        return text_features

def padding_resize(x, ratios, target_h, target_w):
    n,h,w = x.size(0),target_h, target_w
    ratios = ratios.view(-1)
    need_w = (h * ratios).int()
    need_padding_mask = need_w < w
    pad_left = torch.where(need_padding_mask, (w - need_w) // 2, torch.tensor(0).to(x.device))
    pad_right = torch.where(need_padding_mask, w - need_w - pad_left, torch.tensor(0).to(x.device)).tolist()
    need_w = need_w.tolist()
    pad_left = pad_left.tolist()
    x = torch.concat([F.pad(F.interpolate(x[i:i+1,...], (h, need_w[i]), mode="bilinear", align_corners=False), (pad_left[i], pad_right[i]))  if need_padding_mask[i] else F.interpolate(x[i:i+1,...], (h, need_w[i]), mode="bilinear", align_corners=False)[...,pad_left[i]:pad_left[i]+w]  for i in range(n)], dim=0)
    return x

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

from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from ..modules import BasicConv2d
block_map = {'BasicBlock': BasicBlock,
             'Bottleneck': Bottleneck}

class Pre_ResNet9(ResNet):
    def __init__(self, type, block, channels=[32, 64, 128, 256], in_channel=1, layers=[1, 2, 2, 1], strides=[1, 2, 2, 1], maxpool=True):
        if block in block_map.keys():
            block = block_map[block]
        else:
            raise ValueError(
                "Error type for -block-Cfg-, supported: 'BasicBlock' or 'Bottleneck'.")
        self.maxpool_flag = maxpool
        super(Pre_ResNet9, self).__init__(block, layers)

        # Not used #
        self.fc = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        ############
        self.inplanes = channels[0]
        self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.conv1 = BasicConv2d(in_channel, self.inplanes, 3, 1, 1)

        self.layer1 = self._make_layer(
            block, channels[0], layers[0], stride=strides[0], dilate=False)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        if blocks >= 1:
            layer = super()._make_layer(block, planes, blocks, stride=stride, dilate=dilate)
        else:
            def layer(x): return x
        return layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool_flag:
            x = self.maxpool(x)

        x = self.layer1(x)
        return x
    
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
    
class AttentionFusion(nn.Module): 
    def __init__(self, in_channels, squeeze_ratio, feat_len):
        super(AttentionFusion, self).__init__()
        hidden_dim = int(in_channels / squeeze_ratio)
        self.feat_len = feat_len
        self.conv = SetBlockWrapper(
            nn.Sequential(
                conv1x1(in_channels * feat_len, hidden_dim), 
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU(inplace=True), 
                conv3x3(hidden_dim, hidden_dim), 
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU(inplace=True), 
                conv1x1(hidden_dim, in_channels * feat_len), 
            )
        )
    
    def forward(self, feat_list): 
        '''
            sil_feat: [n, c, s, h, w]
            map_feat: [n, c, s, h, w]
            ...
        '''
        feats = torch.cat(feat_list, dim=1)
        score = self.conv(feats) # [n, 2 * c, s, h, w]
        score = rearrange(score, 'n (c d) s h w -> n c d s h w', d=self.feat_len)
        score = F.softmax(score, dim=2)
        retun = feat_list[0]*score[:,:,0]
        for i in range(1, self.feat_len):
            retun += feat_list[i]*score[:,:,i]
        return retun

class LMGait(BaseModel):
    def build_network(self, model_cfg):
        # get pretained models
        self.pretrained_dinov2 = model_cfg["pretrained_dinov2"]
        self.pretrained_mask_branch = model_cfg["pretrained_mask_branch"]

        # set input size
        self.image_size = model_cfg["image_size"]
        self.sils_size = model_cfg["sils_size"]

        # set feature dim
        self.f4_dim = model_cfg["Mask_Branch"]['source_dim']
        self.fc_dim = self.f4_dim*4
        self.mask_dim = model_cfg["Mask_Branch"]['target_dim']
        self.app_dim = model_cfg["Appearance_Branch"]['target_dim']
        self.denoising_dim = model_cfg["Denoising_Branch"]['target_dim']

        # init submodules
        self.Denoising_Branch = infoDistillation(**model_cfg["Denoising_Branch"])
        self.Appearance_Branch = infoDistillation(**model_cfg["Appearance_Branch"])
        self.Mask_Branch = infoDistillation(**model_cfg["Mask_Branch"])
        self.gait_net = Baseline(model_cfg)

        self.CLIP_Branch = CLIP(**model_cfg["CLIP_Branch"])

        self.partition = model_cfg["dataset_partition"]

        self.text_feature = self.CLIP_Branch.get_text_feature(self.partition ,self.device)

        self.adapter_fc1 = nn.Linear(512,1536)

        self.adapter_time = Adapter2(**model_cfg["Adapter"])
        self.reduce_conv2d = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.expand_conv2d = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        model_cfg['backbone_cfg']['in_channel'] = model_cfg['Denoising_Branch']['target_dim']
        self.pre_part = SetBlockWrapper(Pre_ResNet9(**model_cfg['backbone_cfg']))

        model_cfg['backbone_cfg']['in_channel'] = model_cfg['Appearance_Branch']['target_dim']
        self.pre_rgb = SetBlockWrapper(Pre_ResNet9(**model_cfg['backbone_cfg']))

        self.fusion = AttentionFusion(**model_cfg['AttentionFusion'])

    def init_DINOv2(self):
        self.backbone = vit_small(logger = self.msg_mgr)
        self.msg_mgr.log_info(f'load model from: {self.pretrained_dinov2}')
        pretrain_dict = torch.load(self.pretrained_dinov2)
        msg = self.backbone.load_state_dict(pretrain_dict, strict=True)
        n_parameters = sum(p.numel() for p in self.backbone.parameters())
        self.msg_mgr.log_info('Missing keys: {}'.format(msg.missing_keys))
        self.msg_mgr.log_info('Unexpected keys: {}'.format(msg.unexpected_keys))
        self.msg_mgr.log_info(f"=> loaded successfully '{self.pretrained_dinov2}'")
        self.msg_mgr.log_info('DINOv2 Count: {:.5f}M'.format(n_parameters / 1e6))

    def init_Mask_Branch(self):
        self.msg_mgr.log_info(f'load model from: {self.pretrained_mask_branch}')
        load_dict = torch.load(self.pretrained_mask_branch, map_location=torch.device("cpu"))['model']
        msg = self.Mask_Branch.load_state_dict(load_dict, strict=True)
        n_parameters = sum(p.numel() for p in self.Mask_Branch.parameters())
        self.msg_mgr.log_info('Missing keys: {}'.format(msg.missing_keys))
        self.msg_mgr.log_info('Unexpected keys: {}'.format(msg.unexpected_keys))
        self.msg_mgr.log_info(f"=> loaded successfully '{self.pretrained_mask_branch}'")
        self.msg_mgr.log_info('SegmentationBranch Count: {:.5f}M'.format(n_parameters / 1e6))

    def init_CLIP_Branch(self):
        self.CLIP_Branch.initialize_parameters()
        self.msg_mgr.log_info(f'load model from: {CLIP_VIT_B16_PATH}')
        checkpoint = torch.jit.load(CLIP_VIT_B16_PATH, map_location='cpu')
        assert CLIP_VIT_B16_PATH is not None, \
            'Please set CLIP_VIT_B16_PATH in configs.py'
        self.CLIP_Branch.load_state_dict(checkpoint.state_dict(), strict=False)
        copy_weights(self.CLIP_Branch.transformer.resblocks[11], self.CLIP_Branch.transformer.resblocks[12])
        convert_weights(self.CLIP_Branch)
        # init text encoder
        for k,v in self.CLIP_Branch.named_parameters():
            if 'adapter' in k:
                v.requires_grad = True
                v.data = v.data.float()
                # if '12' in k:
                #     print(k)
            else:
                v.requires_grad = False
        

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

        n_parameters = sum(p.numel() for p in self.parameters())
        self.msg_mgr.log_info('Expect backbone Count: {:.5f}M'.format(n_parameters / 1e6))
        
        self.init_CLIP_Branch()
        self.init_DINOv2()
        self.backbone.eval()
        self.backbone.requires_grad_(False)
        
        self.Mask_Branch.train()
        self.Mask_Branch.requires_grad_(True)
        
        n_parameters = sum(p.numel() for p in self.parameters())
        self.msg_mgr.log_info('All Backbone Count: {:.5f}M'.format(n_parameters / 1e6))
        self.msg_mgr.log_info("=> init successfully")

    # resize image
    def preprocess(self, sils, image_size, mode='bilinear'):
        # shape: [nxs,c,h,w] / [nxs,c,224,112]
        return F.interpolate(sils, (image_size*2, image_size), mode=mode, align_corners=False)

    def min_max_norm(self, x):
        return (x - x.min())/(x.max() - x.min())

    # cal foreground
    def get_body(self, mask):
        # value: [0,1]  shape: [nxs, h, w, c]
        def judge_edge(image, edge=1):
            # [nxs,h,w]
            edge_pixel_count = image[:, :edge, :].sum(dim=(1,2)) + image[:, -edge:, :].sum(dim=(1,2))
            return edge_pixel_count > (image.size(2)) * edge
        condition_mask = torch.round(mask[...,0]) - mask[...,0].detach() + mask[...,0]
        condition_mask = judge_edge(condition_mask, 5)
        mask[condition_mask, :, :, 0] = mask[condition_mask, :, :, 1]
        return mask[...,0]
    
    def connect_loss(self, images, n, s, c):
        images = images.view(n*s,c,self.sils_size*2,self.sils_size)
        gradient_x = F.conv2d(images, torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])[None,None,...].repeat(1,c,1,1).to(images.dtype).to(images.device), padding=1)
        gradient_y = F.conv2d(images, torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])[None,None,...].repeat(1,c,1,1).to(images.dtype).to(images.device), padding=1)
        loss_connectivity = (torch.sum(torch.abs(gradient_x)) + torch.sum(torch.abs(gradient_y))) / (n*s*c*self.sils_size*2*self.sils_size)
        return loss_connectivity
    
    # Binarization and Closing operations to enhance foreground
    def get_edge(self, sils, threshold=1):
        mask_sils = torch.round(sils * threshold)
        kernel = torch.ones((3,3))
        dilated_mask = morph.dilation(mask_sils, kernel.to(sils.device)).detach()  # Dilation
        kernel = torch.ones((5,5))
        eroded_mask = morph.erosion(dilated_mask, kernel.to(sils.device)).detach()  # Erosion
        edge_mask = (dilated_mask > 0.5) ^ (eroded_mask > 0.5)
        sils = edge_mask * sils + (eroded_mask > 0.5) * torch.ones_like(sils, dtype=sils.dtype, device=sils.device)
        return sils

    def diversity_loss(self, images, max_p):
        # [ns, hw, c]
        p = torch.sum(images, dim=1) / (torch.sum(images, dim=(1,2)) + 1e-6).view(-1,1).repeat(1,max_p)
        entropies = -torch.sum(p * torch.log2(p + 1e-6), dim=1)
        max_p = torch.Tensor([1/max_p]).repeat(max_p).to(images.dtype).to(images.device)
        max_entropies = -torch.sum(max_p * torch.log2(max_p), dim=0)
        return torch.mean(max_entropies - entropies)

    def forward(self, inputs):
        if self.training:
            if self.iteration == 500 and '.pt' in self.pretrained_mask_branch:
                self.init_Mask_Branch()
            if self.iteration >= 500:
                self.Mask_Branch.eval()
                self.Mask_Branch.requires_grad_(False)

        ipts, labs, ty, vi, seqL = inputs
        sils = ipts[0]                      # input_images;         shape: [n,s,c,h,w];
        ratios = ipts[1]                    # real_image_ratios     shape: [n,s,ratio];     ratio: w/h,  e.g. 112/224=0.5;
        del ipts

        text_feature = self.CLIP_Branch(self.text_feature)
        text_feature = self.adapter_fc1(text_feature)

        with torch.no_grad():
            n,s,c,h,w = sils.size()
            sils = rearrange(sils, 'n s c h w -> (n s) c h w').contiguous()
            if h == 2*w:
                outs = self.preprocess(sils, self.image_size)                                           # [ns,c,448,224]    if have used pad_resize for input images
            else:
                outs = self.preprocess(padding_resize(sils, ratios, 256, 128), self.image_size)         # [ns,c,448,224]    if have not used pad_resize for input images
            outs = self.backbone(outs, is_training=True) # [ns,h*w,c]
        
        outs_last1 = outs["x_norm_patchtokens"].contiguous()
        outs_last4 = outs["x_norm_patchtokens_mid4"].contiguous()

        outs_last4 = outs_last4 + text_feature.unsqueeze(0)

        outs_last1 = rearrange(outs_last1.view(n, s, self.image_size//7, self.image_size//14, -1), 'n s h w c -> (n s) c h w').contiguous()
        outs_last4 = rearrange(outs_last4.view(n, s, self.image_size//7, self.image_size//14, -1), 'n s h w c -> (n s) c h w').contiguous()
        outs_last1 = self.preprocess(outs_last1, self.sils_size) # [ns,c,64,32]
        outs_last4 = self.preprocess(outs_last4, self.sils_size) # [ns,c,64,32]
        outs_last1 = rearrange(outs_last1.view(n, s, -1, self.sils_size*2, self.sils_size), 'n s c h w -> (n s) (h w) c').contiguous()
        outs_last4 = rearrange(outs_last4.view(n, s, -1, self.sils_size*2, self.sils_size), 'n s c h w -> (n s) (h w) c').contiguous()

        # get foreground
        mask = torch.ones_like(outs_last1[...,0], device=outs_last1.device, dtype=outs_last1.dtype).view(n*s,1,self.sils_size*2,self.sils_size)
        mask = padding_resize(mask, ratios, self.sils_size*2, self.sils_size)
        foreground = outs_last1.view(-1, self.f4_dim)[mask.view(-1) != 0]
        fore_feat, loss_mse1 = self.Mask_Branch(foreground)
        foreground = torch.zeros_like(mask, dtype=fore_feat.dtype, device=fore_feat.device).view(-1,1).repeat(1,self.mask_dim)
        foreground[mask.view(-1) != 0] = fore_feat
        loss_connectivity_shape = self.connect_loss(foreground, n, s, self.mask_dim)
        foreground = foreground.detach().clone()
        foreground = self.get_body(foreground.view(n*s,self.sils_size*2,self.sils_size,self.mask_dim)).view(n*s,-1) # [n*s,h*w]
        foreground = self.get_edge(foreground.view(n*s,1,self.sils_size*2,self.sils_size)).view(n*s,-1) # [n*s,h*w]
        del fore_feat, mask

        # get denosing
        denosing = outs_last4.view(-1, self.fc_dim)[foreground.view(-1) != 0]
        den_feat, _ = self.Denoising_Branch(denosing)
        denosing = torch.zeros_like(foreground, dtype=den_feat.dtype, device=den_feat.device).view(-1,1).repeat(1,self.denoising_dim)
        denosing[foreground.view(-1) != 0] = den_feat
        loss_connectivity_part = self.connect_loss(denosing.view(n*s,-1,self.denoising_dim)[...,:-1].permute(0,2,1), n, s, (self.denoising_dim-1))
        loss_diversity_part = self.diversity_loss(denosing.view(n*s,-1,self.denoising_dim), self.denoising_dim)
        del den_feat

        # get appearance
        appearance = outs_last4.view(-1, self.fc_dim)[foreground.view(-1) != 0]
        app_feat, _ = self.Appearance_Branch(appearance)
        appearance = torch.zeros_like(foreground, dtype=app_feat.dtype, device=app_feat.device).view(-1,1).repeat(1,self.app_dim)
        appearance[foreground.view(-1) != 0] = app_feat
        appearance = appearance.view(n*s,-1,self.app_dim)
        del app_feat

        # vis
        if self.training:
            try:
                vis_num = min(5, n*s)
                vis_mask = foreground.view(n*s, self.sils_size*2*self.sils_size, -1)[:vis_num].detach().cpu().numpy()
                vis_denosing = pca_image(data={'embeddings':denosing.view(n*s, self.sils_size*2*self.sils_size, -1)[:vis_num].detach().cpu().numpy()}, mask=vis_mask, root=None, model_name=None, dataset=None, n_components=3, is_return=True) # n s c h w
                vis_appearance = pca_image(data={'embeddings':appearance.view(n*s, self.sils_size*2*self.sils_size, -1)[:vis_num].detach().cpu().numpy()}, mask=vis_mask, root=None, model_name=None, dataset=None, n_components=3, is_return=True) # n s c h w
            except:
                vis_denosing = torch.ones_like(foreground).view(n,s,1,self.sils_size*2,self.sils_size).detach().cpu().numpy()
                vis_appearance = torch.ones_like(foreground).view(n,s,1,self.sils_size*2,self.sils_size).detach().cpu().numpy()

        # # Black DA
        # if self.training:
        #     mask_idx = random.sample(list(range(n)), int(round(n*0.2)))
        #     feat_list = [denosing.view(n,s,-1), appearance.view(n,s,-1)]
        #     for i in mask_idx:
        #         idx = random.sample(list(range(2)), 1)
        #         for j in idx:
        #             feat_list[j][i] = torch.zeros_like(feat_list[j][i], device=feat_list[j].device, dtype=feat_list[j].dtype)

        denosing = self.pre_part(denosing.view(n,s,self.sils_size*2,self.sils_size,self.denoising_dim).permute(0, 4, 1, 2, 3).contiguous())  # [n, c, s, h, w]
        appearance = self.pre_rgb(appearance.view(n,s,self.sils_size*2,self.sils_size,self.app_dim).permute(0, 4, 1, 2, 3).contiguous())  # [n, c, s, h, w]
        outputs = self.fusion([denosing, appearance])

        out_origin = outputs

        B_, C_, T_, H_, W_ = outputs.shape
        outputs = outputs.permute(0, 2, 1, 3, 4).contiguous() #B,T,C,H,W
        outputs = outputs.view(B_*T_, C_, H_, W_) 

        out_channel = 64

        outputs = self.reduce_conv2d(outputs) 
        outputs = outputs.view(B_, T_, out_channel, H_, W_).permute(0, 1, 3, 4, 2).contiguous()

        outputs = outputs.view(B_ * T_, H_ * W_, out_channel)  # BT,HW,C

        outputs = self.adapter_time(outputs, B=B_)

        outputs = outputs.view(B_*T_, H_, W_, out_channel)        # [B*T, H, W, C]
        outputs = outputs.permute(0, 3, 1, 2).contiguous()  # [B*T, C, H, W]

        outputs = self.expand_conv2d(outputs)         # [B*T, 64, H, W]
        outputs = outputs.view(B_, T_, C_, H_, W_).permute(0, 2, 1, 3, 4).contiguous()

        outputs = outputs + out_origin

        # get embeding
        embed_1, logits = self.gait_net(
                                        outputs,
                                        seqL,
                                        )

        if self.training:
            retval = {
                'training_feat': {
                    'shape_connect':loss_connectivity_shape*0.02,
                    'shape_mse': loss_mse1,
                    'part_connect':loss_connectivity_part*0.01,
                    'part_diversity':loss_diversity_part*5,
                    'triplet': {'embeddings': embed_1, 'labels': labs},
                    'softmax': {'logits': logits, 'labels': labs},
                },
                'visual_summary': {
                    'image/input': sils.view(n*s, c, h, w),
                    'image/foreground': self.min_max_norm(rearrange(foreground.view(n, s, self.sils_size*2, self.sils_size, -1), 'n s h w c -> (n s) c h w').contiguous()),
                    'image/denosing':self.min_max_norm(rearrange(torch.from_numpy(vis_denosing).float(), 'n s c h w -> (n s) c h w').contiguous()),
                    'image/appearance': self.min_max_norm(rearrange(torch.from_numpy(vis_appearance).float(), 'n s c h w -> (n s) c h w').contiguous()),
                },
                'inference_feat': {
                    'embeddings': embed_1
                }
            }
        else:
            retval = {
                'training_feat': {},
                'visual_summary': {},
                'inference_feat': {'embeddings': embed_1}
            }
        return retval
