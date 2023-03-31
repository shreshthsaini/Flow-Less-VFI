# Functions are imported and modified from https://github.com/JingyunLiang/VRT/blob/94a5f504eb84aedf1314de5389f45f4ba1c2d022/models/network_vrt.py

import numpy as np
import math
from functools import reduce, lru_cache
from typing import Type, Callable, Tuple, Optional, Set, List, Union

import torch,timm
from torch import nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
from torchinfo import summary

import custom_layers
import blocks

import warnings
warnings.filterwarnings("ignore")


class Modified_VRT_Config0(nn.Module):
	"""
	Modified Video Restoration Transformer Config0.
	Args:

	"""
	def __init__(self
	) -> None:
		super().__init__()

		# Architecture
		self.conv3d_first = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(1,3,3),stride=(1,1,1),padding=(0,1,1))

		self.stage1 = blocks.Stage(
			in_channels=64,
			dim=128,
			depth=2,
			num_heads=4,
			window_size=[3,8,8],
			qkv_bias=True,
			qk_scale=None,
			mul_attn_ratio=0.75,
			mlp_ratio=2.0,
			drop_path=0.0,
			act_layer=nn.GELU,
			norm_layer=nn.LayerNorm,
			reshape="down"
		)

		self.stage2 = blocks.Stage(
			in_channels=128,
			dim=256,
			depth=4,
			num_heads=4,
			window_size=[3,8,8],
			qkv_bias=True,
			qk_scale=None,
			mul_attn_ratio=0.75,
			mlp_ratio=2.0,
			drop_path=0.0,
			act_layer=nn.GELU,
			norm_layer=nn.LayerNorm,
			reshape="down"
		)

		self.stage3 = blocks.Stage(
			in_channels=256,
			dim=256,
			depth=4,
			num_heads=8,
			window_size=[3,8,8],
			qkv_bias=True,
			qk_scale=None,
			mul_attn_ratio=0.75,
			mlp_ratio=2.0,
			drop_path=0.0,
			act_layer=nn.GELU,
			norm_layer=nn.LayerNorm,
			reshape="down"
		)

		self.stage4 = blocks.Stage(
			in_channels=256,
			dim=256,
			depth=8,
			num_heads=8,
			window_size=[3,8,8],
			qkv_bias=True,
			qk_scale=None,
			mul_attn_ratio=0.75,
			mlp_ratio=2.0,
			drop_path=0.0,
			act_layer=nn.GELU,
			norm_layer=nn.LayerNorm,
			reshape="none"
		)

		self.stage5 = blocks.Stage(
			in_channels=256,
			dim=256,
			depth=4,
			num_heads=8,
			window_size=[3,8,8],
			qkv_bias=True,
			qk_scale=None,
			mul_attn_ratio=0.75,
			mlp_ratio=2.0,
			drop_path=0.0,
			act_layer=nn.GELU,
			norm_layer=nn.LayerNorm,
			reshape="up"
		)

		self.stage6 = blocks.Stage(
			in_channels=512,
			dim=128,
			depth=4,
			num_heads=4,
			window_size=[3,8,8],
			qkv_bias=True,
			qk_scale=None,
			mul_attn_ratio=0.75,
			mlp_ratio=2.0,
			drop_path=0.0,
			act_layer=nn.GELU,
			norm_layer=nn.LayerNorm,
			reshape="up"
		)

		self.stage7 = blocks.Stage(
			in_channels=256,
			dim=64,
			depth=2,
			num_heads=4,
			window_size=[3,8,8],
			qkv_bias=True,
			qk_scale=None,
			mul_attn_ratio=0.75,
			mlp_ratio=2.0,
			drop_path=0.0,
			act_layer=nn.GELU,
			norm_layer=nn.LayerNorm,
			reshape="up"
		)

		self.reflection_pad2d = custom_layers.reflection_pad2d
		self.linear_fuse = nn.Conv2d(in_channels=4*64, out_channels=64, kernel_size=(1,1), stride=(1,1))
		self.conv2d_last = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=(7,7),stride=(1,1))


	def forward(self, x):
		"""
		Args:
			x (torch.Tensor): The video input of shape (B,D,C,H,W) where B: batch_size, C: channels, D: no.of frames/depth, H: height of the video, W: width of the video.
		"""
		x_mean = x.mean([1,3,4], keepdim=True)

		x = x - x_mean
		x = x.transpose(1,2)
		x = self.conv3d_first(x)

		x1 = self.stage1(x)
		x2 = self.stage2(x1)
		x3 = self.stage3(x2)
		y3 = self.stage4(x3)
		y2 = self.stage5(y3)
		y1 = self.stage6(torch.concat((y2,x2), dim=1))
		x = self.stage7(torch.concat((y1,x1), dim=1))

		x = torch.cat(torch.unbind(x, 2), 1)
		x = F.leaky_relu(self.linear_fuse(x), 0.2)
		x = self.reflection_pad2d(x, pad=3)
		x = self.conv2d_last(x)
		x = torch.stack(torch.split(x, dim=1, split_size_or_sections=3), 1)
		x = x + x_mean
		x = x[:,0]

		return x

# model = Modified_VRT_Config0()
# summary(model, input_data=torch.randn(6, 4, 3, 64, 64), col_names=("input_size","output_size"))
