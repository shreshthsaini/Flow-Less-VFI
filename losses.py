import torch,timm,torchvision
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import SoftTargetCrossEntropy
    
# Perceptual Convergence Loss
class PerceptualConvergenceLoss(nn.Module):
	def __init__(self,
		feature_layers=[0, 1, 2, 3],
		style_layers=[]
	) -> None:
		"""
		VGG16 Perceptual Loss with for Real-Time Style Transfer and Super-Resolution.
		Code from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
		Loss for convergence of prediction to both frames to target-frame during supervised learning i.e with target as reference. Loss includes MSE and VGG16 Perceptual-Loss.
		"""
		super().__init__()

		# VGG16 Loss
		blocks = []
		blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
		blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
		blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
		blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
		for bl in blocks:
			for p in bl.parameters():
				p.requires_grad = False
		self.blocks = torch.nn.ModuleList(blocks).requires_grad_(False)
		self.transform = torch.nn.functional.interpolate
		self.resize = True
		self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
		self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
		self.feature_layers = feature_layers
		self.style_layers = style_layers

		# MSE Loss
		self.MSE_Loss = torch.nn.MSELoss()

	def VGG16_Loss(self, input, target):
		if input.shape[1] != 3:
			input = input.repeat(1, 3, 1, 1)
			target = target.repeat(1, 3, 1, 1)
		input = (input-self.mean) / self.std
		target = (target-self.mean) / self.std
		if self.resize:
			input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
			target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
		loss = 0.0
		x = input
		y = target
		for i, block in enumerate(self.blocks):
			x = block(x)
			y = block(y)
			if i in self.feature_layers:
				loss += torch.nn.functional.l1_loss(x, y)
			if i in self.style_layers:
				act_x = x.reshape(x.shape[0], x.shape[1], -1)
				act_y = y.reshape(y.shape[0], y.shape[1], -1)
				gram_x = act_x @ act_x.permute(0, 2, 1)
				gram_y = act_y @ act_y.permute(0, 2, 1)
				loss += torch.nn.functional.l1_loss(gram_x, gram_y)
		return loss
	
	def forward(self, output, target):
		loss = self.MSE_Loss(output, target) + self.VGG16_Loss(output, target)
		return loss
