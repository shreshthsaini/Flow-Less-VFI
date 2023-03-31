# Functions are imported and modified from https://github.com/JingyunLiang/VRT/blob/94a5f504eb84aedf1314de5389f45f4ba1c2d022/models/network_vrt.py

import numpy as np
import math
from functools import reduce, lru_cache
from typing import Type, Callable, Tuple, Optional, Set, List, Union

import torch,timm
from torch import nn
import einops
from einops.layers.torch import Rearrange

import custom_layers
import attentions


# Temporal Mutual Self-Attention Group
class TMSAG(nn.Module):
	"""
	Temporal Mutual Self Attention Group (TMSAG).
	Args:
		input_resolution (tuple[int]): Input resolution.
		dim (int): Number of feature channels.
		depth (int): Depths of this stage.
		num_heads (int): Number of attention heads.
		window_size (tuple[int]): (temporal_length, height, width) Dimensions of the window. Generally height = width = window_size. (default: (6,8,8))
		shift_size (tuple[int]): (temporal_shift, height_shift, width_shift) Shift size for Mutual-Attention and Self-Attention. (default: (0,0,0))
		qkv_bias (boolean):  If True, add a learnable bias to query, key, value. (default: True)
		qk_scale (float): The qk scale coefficient. The default value is head_dim ** -0.5.
		mut_attn (bool): If True, use mutual and self attention. (default: True)
		mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
		drop_path (float|List[float]): Stochastic depth rate. (default: 0.0)
		act_layer (torch.nn.Module): Activation layer. (default: nn.GELU)
		norm_layer (nn.Module): Normalization layer. (default: nn.LayerNorm)
	"""

	def __init__(self,
		# input_resolution,
		dim,
		depth,
		num_heads,
		window_size=[6, 8, 8],
		shift_size=None,
		qkv_bias=False,
		qk_scale=None,
		mut_attn=True,
		mlp_ratio=2.,
		drop_path=0.,
		act_layer=nn.GELU,
		norm_layer=nn.LayerNorm
	):
		super().__init__()
		# self.input_resolution = input_resolution
		self.window_size = window_size
		self.shift_size = list(i // 2 for i in window_size) if shift_size is None else shift_size

		# [TMSA]xN blocks
		self.blocks = nn.ModuleList([
			attentions.TMSA(
				# input_resolution=input_resolution,
				dim=dim,
				num_heads=num_heads,
				window_size=window_size,
				shift_size=[0, 0, 0] if i % 2 == 0 else self.shift_size,
				qkv_bias=qkv_bias,
				qk_scale=qk_scale,
				mut_attn=mut_attn,
				mlp_ratio=mlp_ratio,
				drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
				act_layer=act_layer,
				norm_layer=norm_layer
			)
			for i in range(depth)])

	def forward(self, x):
		""" Forward function.
		Args:
			x: Input feature, tensor size (B, C, D, H, W).
		"""
		B, C, D, H, W = x.shape
		window_size, shift_size = custom_layers.get_window_size((D, H, W), self.window_size, self.shift_size)
		x = einops.rearrange(x, 'b c d h w -> b d h w c')
		Dp = int(np.ceil(D / window_size[0])) * window_size[0]
		Hp = int(np.ceil(H / window_size[1])) * window_size[1]
		Wp = int(np.ceil(W / window_size[2])) * window_size[2]
		attn_mask = custom_layers.compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

		for blk in self.blocks:
			x = blk(x, attn_mask)

		x = x.view(B, D, H, W, -1)
		x = einops.rearrange(x, 'b d h w c -> b c d h w')

		return x
	

# Residual Temporal Mutual Self-Attention
class RTMSA(nn.Module):
	"""
	Residual Temporal Mutual Self Attention (RTMSA). Only used in stage 8.
	Args:
		# input_resolution (tuple[int]): Input resolution.
		dim (int): Number of feature channels.
		depth (int): Depths of this stage.
		num_heads (int): Number of attention heads.
		window_size (tuple[int]): (temporal_length, height, width) Dimensions of the window. Generally height = width = window_size. (default: (6,8,8))
		qkv_bias (boolean):  If True, add a learnable bias to query, key, value. (default: True)
		qk_scale (float): The qk scale coefficient. The default value is head_dim ** -0.5.
		mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
		drop_path (float|List[float]): Stochastic depth rate. (default: 0.0)
		act_layer (torch.nn.Module): Activation layer. (default: nn.GELU)
		norm_layer (nn.Module): Normalization layer. (default: nn.LayerNorm)
	"""

	def __init__(self,
		# input_resolution,
		dim,
		depth,
		num_heads,
		window_size,
		qkv_bias=True,
		qk_scale=None,
		mlp_ratio=2.,
		drop_path=0.,
		act_layer=nn.GELU,
		norm_layer=nn.LayerNorm
	):
		super(RTMSA, self).__init__()
		self.dim = dim
		# self.input_resolution = input_resolution

		self.residual_group = TMSAG(
									# input_resolution=input_resolution,
									dim=dim,
									depth=depth,
									num_heads=num_heads,
									window_size=window_size,
									shift_size=None,
									qkv_bias=qkv_bias,
									qk_scale=qk_scale,
									mut_attn=False,
									mlp_ratio=mlp_ratio,
									drop_path=drop_path,
									act_layer=act_layer,
									norm_layer=norm_layer
								)

		self.linear = nn.Linear(dim, dim)

	def forward(self, x):
		return x + self.linear(self.residual_group(x).transpose(1, 4)).transpose(1, 4)
	

# Stage of VRT
class Stage(nn.Module):
	"""
	Residual Temporal Mutual Self Attention Group and Parallel Warping.
	Args:
		in_channels (int): Number of input channels.
		# input_resolution (tuple[int]): Input resolution.
		dim (int): Number of channels.
		depth (int): Number of blocks.
		num_heads (int): Number of attention heads.
		window_size (tuple[int]): (temporal_length, height, width) Dimensions of the window. Generally height = width = window_size.
		qkv_bias (boolean):  If True, add a learnable bias to query, key, value. (default: True)
		qk_scale (float): The qk scale coefficient. The default value is head_dim ** -0.5.
		mul_attn_ratio (float): Ratio of mutual attention layers. (default: 0.75)
		mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
		drop_path (float|List[float]): Stochastic depth rate. (default: 0.0)
		act_layer (torch.nn.Module): Activation layer. (default: nn.GELU)
		norm_layer (nn.Module): Normalization layer. (default: nn.LayerNorm)
		reshape (str): Downscale (down), upscale (up) or keep the size (none).
	"""

	def __init__(self,
		in_channels,
		# input_resolution,
		dim,
		depth,
		num_heads,
		window_size,
		qkv_bias=True,
		qk_scale=None,
		mul_attn_ratio=0.75,
		mlp_ratio=2.,
		drop_path=0.,
		act_layer=nn.GELU,
		norm_layer=nn.LayerNorm,
		reshape=None
	):
		super(Stage, self).__init__()

		# Reshape the Tensor
		if reshape == 'none':
			self.reshape = nn.Sequential(
				Rearrange('n c d h w -> n d h w c'),
				nn.LayerNorm(dim),
				Rearrange('n d h w c -> n c d h w')
			)
		elif reshape == 'down':
			self.reshape = nn.Sequential(
				Rearrange('n c d (h neih) (w neiw) -> n d h w (neiw neih c)', neih=2, neiw=2),
				nn.LayerNorm(4 * in_channels), nn.Linear(4 * in_channels, dim),
				Rearrange('n d h w c -> n c d h w')
			)
		elif reshape == 'up':
			self.reshape = nn.Sequential(
				Rearrange('n (neiw neih c) d h w -> n d (h neih) (w neiw) c', neih=2, neiw=2),
				nn.LayerNorm(in_channels // 4), nn.Linear(in_channels // 4, dim),
				Rearrange('n d h w c -> n c d h w')
			)

		# Mutual-Attention and Self-Attention
		self.residual_group1 = TMSAG(
									# input_resolution=input_resolution,
									dim=dim,
									depth=int(depth * mul_attn_ratio),
									num_heads=num_heads,
									window_size=(2, window_size[1], window_size[2]),
									shift_size=None,
									qkv_bias=qkv_bias,
									qk_scale=qk_scale,
									mut_attn=True,
									mlp_ratio=mlp_ratio,
									drop_path=drop_path,
									act_layer=act_layer,
									norm_layer=norm_layer
								)
		self.linear1 = nn.Linear(dim, dim)

		# Self-Attention
		self.residual_group2 = TMSAG(
									# input_resolution=input_resolution,
									dim=dim,
									depth=depth - int(depth * mul_attn_ratio),
									num_heads=num_heads,
									window_size=window_size,
									qkv_bias=qkv_bias,
									qk_scale=qk_scale,
									mut_attn=False,
									mlp_ratio=mlp_ratio,
									drop_path=drop_path,
									act_layer=act_layer,
									norm_layer=norm_layer
								)
		self.linear2 = nn.Linear(dim, dim)

	def forward(self, x):
		x = self.reshape(x)
		x = self.linear1(self.residual_group1(x).transpose(1, 4)).transpose(1, 4) + x
		x = self.linear2(self.residual_group2(x).transpose(1, 4)).transpose(1, 4) + x

		return x
