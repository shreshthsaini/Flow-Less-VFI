# Functions are imported and modified from https://github.com/JingyunLiang/VRT/blob/94a5f504eb84aedf1314de5389f45f4ba1c2d022/models/network_vrt.py

import numpy as np
import math
from functools import reduce, lru_cache
from typing import Type, Callable, Tuple, Optional, Set, List, Union

import torch,timm
from torch import nn
import torch.nn.functional as F

import custom_layers


# Window Attention Module with Mutual-Attention and Self-Attenion.
class WindowAttention(nn.Module):
	"""
	Window based Multi-Head Mutual-Attention and Self-Attention.
	Args:
		dim (int): Number of input channels.
		num_heads (int): Number of attention heads.
		window_size (tuple[int]): (temporal_length, height, width) Dimensions of the window. Generally height = width = window_size.
		qkv_bias (boolean):  If True, add a learnable bias to query, key, value. (default: True)
		qk_scale (float): The qk scale coefficient. The default value is head_dim ** -0.5.
		mut_attn (boolean): If True, add mutual attention to the module. (default: True)
	"""
	def __init__(self, 
	    dim,
	    num_heads, 
		window_size,
		qkv_bias=False, 
		qk_scale=None, 
		mut_attn=True
	):
		super().__init__()
		self.dim = dim
		self.window_size = window_size
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = qk_scale or head_dim ** -0.5
		self.mut_attn = mut_attn

		# Self-Attention with Relative-Position Bias
		self.relative_position_bias_table = nn.Parameter(torch.zeros((2*window_size[0] - 1)*(2*window_size[1] - 1) * (2*window_size[2] - 1), num_heads))	# (2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH)
		self.register_buffer("relative_position_index", self.get_position_index(window_size))
		custom_layers.trunc_normal_(self.relative_position_bias_table, std=.02)
		
		# Projection Weights
		self.qkv_self = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.proj = nn.Linear(dim, dim)

		# Softmax
		self.softmax = nn.Softmax(dim=-1)

		# Mutual-Attention with Sine-Position Encoding
		if self.mut_attn:
			self.register_buffer("position_bias", self.get_sine_position_encoding(window_size[1:], dim//2, normalize=True))
			self.qkv_mut = nn.Linear(dim, dim * 3, bias=qkv_bias)
			self.proj = nn.Linear(2 * dim, dim)


	def get_position_index(self, window_size):
		"""
		Get pair-wise relative position index for each token inside the window.
		"""
		coords_d = torch.arange(window_size[0])
		coords_h = torch.arange(window_size[1])
		coords_w = torch.arange(window_size[2])
		coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # (3, Wd, Wh, Ww)
		coords_flatten = torch.flatten(coords, 1)  # (3, Wd*Wh*Ww)
		relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (3, Wd*Wh*Ww, Wd*Wh*Ww)
		relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (Wd*Wh*Ww, Wd*Wh*Ww, 3)
		relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
		relative_coords[:, :, 1] += window_size[1] - 1
		relative_coords[:, :, 2] += window_size[2] - 1

		relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
		relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
		relative_position_index = relative_coords.sum(-1)  # (Wd*Wh*Ww, Wd*Wh*Ww)

		return relative_position_index


	def get_sine_position_encoding(self, HW, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
		"""
		Get Sine Position Encoding.
		"""
		if scale is not None and normalize is False:
			raise ValueError("Normalize should be True if scale is passed")

		if scale is None:
			scale = 2 * math.pi

		not_mask = torch.ones([1, HW[0], HW[1]])
		y_embed = not_mask.cumsum(1, dtype=torch.float32)
		x_embed = not_mask.cumsum(2, dtype=torch.float32)
		if normalize:
			eps = 1e-6
			y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
			x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

		dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
		dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

		# BxCxHxW
		pos_x = x_embed[:, :, :, None] / dim_t
		pos_y = y_embed[:, :, :, None] / dim_t
		pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
		pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
		pos_embed = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

		return pos_embed.flatten(2).permute(0, 2, 1).contiguous()


	def attention(self, q, k, v, mask, x_shape, relative_position_encoding=True):
		B_, N, C = x_shape
		attn = (q * self.scale) @ k.transpose(-2, -1)

		if relative_position_encoding:
			relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)  # (Wd*Wh*Ww, Wd*Wh*Ww, nH)
			attn = attn + relative_position_bias.permute(2, 0, 1).unsqueeze(0)  # B_, nH, N, N

		if mask is None:
			attn = self.softmax(attn)
		else:
			nW = mask.shape[0]
			attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask[:, :N, :N].unsqueeze(1).unsqueeze(0)
			attn = attn.view(-1, self.num_heads, N, N)
			attn = self.softmax(attn)

		x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

		return x
	
	def forward(self, x, mask=None):
		"""
		Forward function.
		Args:
			x: input features with shape of (num_windows*B, N, C)
			mask: (0/-inf) mask with shape of (num_windows, N, N) or None
		"""

		# Self-Attention
		B_, N, C = x.shape
		qkv = self.qkv_self(x).reshape(B_, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

		# Multi-Head Self-Attenton
		x_out = self.attention(q, k, v, mask, (B_, N, C), relative_position_encoding=True)

		# Mutual-Attention
		if self.mut_attn:
			qkv = self.qkv_mut(x + self.position_bias.repeat(1, 2, 1)).reshape(B_, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4) # (B_, N, 3, H, C//H) -> # (3, B_, H, N, C//H)
			(q1, q2), (k1, k2), (v1, v2) = torch.chunk(qkv[0], 2, dim=2), torch.chunk(qkv[1], 2, dim=2), torch.chunk(qkv[2], 2, dim=2)  # B_, nH, N/2, C
			x1_aligned = self.attention(q2, k1, v1, mask, (B_, N // 2, C), relative_position_encoding=False)
			x2_aligned = self.attention(q1, k2, v2, mask, (B_, N // 2, C), relative_position_encoding=False)
			x_out = torch.cat([torch.cat([x1_aligned, x2_aligned], 1), x_out], 2)

		# Projection
		x = self.proj(x_out)

		return x


# Temporal Mutual Self-Attention
class TMSA(nn.Module):
	"""
	Temporal Mutual Self Attention (TMSA).
	Args:
		# input_resolution (tuple[int]): Input resolution.
		dim (int): Number of input channels.
		num_heads (int): Number of attention heads.
		window_size (tuple[int]): (temporal_length, height, width) Dimensions of the window. Generally height = width = window_size. (default: (6,8,8))
		shift_size (tuple[int]): (temporal_shift, height_shift, width_shift) Shift size for Mutual-Attention and Self-Attention. (default: (0,0,0))
		qkv_bias (boolean):  If True, add a learnable bias to query, key, value. (default: True)
		qk_scale (float): The qk scale coefficient. The default value is head_dim ** -0.5.
		mut_attn (bool): If True, use mutual and self attention. (default: True)
		mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
		drop_path (float): Stochastic depth rate. (default: 0.0)
		act_layer (torch.nn.Module): Activation layer. (default: nn.GELU)
		norm_layer (torch.nn.Module): Normalization layer. (default: nn.LayerNorm)
	"""

	def __init__(self,
		# input_resolution,
		dim,
		num_heads,
		window_size=(6, 8, 8),
		shift_size=(0, 0, 0),
		qkv_bias=True,
		qk_scale=None,
		mut_attn=True,
		mlp_ratio=2.0,
		drop_path=0.,
		act_layer=nn.GELU,
		norm_layer=nn.LayerNorm
	):
		super().__init__()
		# self.input_resolution = input_resolution
		self.dim = dim
		self.num_heads = num_heads
		self.window_size = window_size
		self.shift_size = shift_size

		assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size[0] must in [0,window_size[0])"
		assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size[1] must in [0,window_size[1])"
		assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size[2] must in [0,window_size[2])"

		self.norm1 = norm_layer(dim)
		self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, mut_attn=mut_attn)
		self.drop_path = custom_layers.DropPath(drop_path) if drop_path > 0. else nn.Identity()
		self.norm2 = norm_layer(dim)
		self.mlp = custom_layers.MLP_GEGLU(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)


	def forward_part1(self, x, mask_matrix):
		B, D, H, W, C = x.shape
		window_size, shift_size = custom_layers.get_window_size((D, H, W), self.window_size, self.shift_size)

		x = self.norm1(x)

		# Pad Feature Maps to multiples of window size
		pad_l = pad_t = pad_d0 = 0
		pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
		pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
		pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
		x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1), mode='constant')

		_, Dp, Hp, Wp, _ = x.shape
		# Cyclic Shift
		if any(i > 0 for i in shift_size):
			shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
			attn_mask = mask_matrix
		else:
			shifted_x = x
			attn_mask = None

		# Partition Windows
		x_windows = custom_layers.window_partition(shifted_x, window_size)	# (B*nW, Wd*Wh*Ww, C)

		# Attention / Shifted Attention
		attn_windows = self.attn(x_windows, mask=attn_mask)	# (B*nW, Wd*Wh*Ww, C)

		# Merge Windows
		attn_windows = attn_windows.view(-1, *(window_size + (C,)))
		shifted_x = custom_layers.window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)	# (B, D', H', W', C)

		# Reverse Cyclic Shift
		if any(i > 0 for i in shift_size):
			x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
		else:
			x = shifted_x

		if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
			x = x[:, :D, :H, :W, :]

		x = self.drop_path(x)

		return x

	def forward_part2(self, x):
		return self.drop_path(self.mlp(self.norm2(x)))

	def forward(self, x, mask_matrix):
		"""
		Forward function.
		Args:
			x: Input feature, tensor size (B, D, H, W, C).
			mask_matrix: Attention mask for cyclic shift.
		"""

		# Attention
		x = x + self.forward_part1(x, mask_matrix)

		# Feed-Forward
		x = x + self.forward_part2(x)

		return x
