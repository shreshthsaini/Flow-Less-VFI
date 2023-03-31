import torch,timm,torchvision
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import SoftTargetCrossEntropy
    

# Self-Supervised Perceptual Convergence Loss
class SelfSupervised_PerceptualConvergenceLoss(nn.Module):
	def __init__(self) -> None:
		"""
		VGG16 Perceptual Loss with for Real-Time Style Transfer and Super-Resolution.
		Code from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
		Loss for convergence of prediction to both frames during self-supervised learning i.e both frames should converge to same prediction. Loss includes MSE and VGG16 Perceptual-Loss.
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

		# MSE Loss
		self.MSELoss1 = torch.nn.MSELoss()

	def VGG16_Loss(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
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
			if i in feature_layers:
				loss += torch.nn.functional.l1_loss(x, y)
			if i in style_layers:
				act_x = x.reshape(x.shape[0], x.shape[1], -1)
				act_y = y.reshape(y.shape[0], y.shape[1], -1)
				gram_x = act_x @ act_x.permute(0, 2, 1)
				gram_y = act_y @ act_y.permute(0, 2, 1)
				loss += torch.nn.functional.l1_loss(gram_x, gram_y)
		return loss
		
	def MSE_Loss(self, output1, output2):
		loss = self.MSELoss1(output1, output2)
		return loss

	def forward(self, output1, output2):
		loss = self.MSE_Loss(output1, output2) + self.VGG16_Loss(output1, output2)
		return loss


# Perceptual Convergence Loss
class PerceptualConvergenceLoss(nn.Module):
	def __init__(self) -> None:
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

		# MSE Loss
		self.MSELoss1 = torch.nn.MSELoss()
		self.MSELoss2 = torch.nn.MSELoss()

	def VGG16_Loss(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
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
			if i in feature_layers:
				loss += torch.nn.functional.l1_loss(x, y)
			if i in style_layers:
				act_x = x.reshape(x.shape[0], x.shape[1], -1)
				act_y = y.reshape(y.shape[0], y.shape[1], -1)
				gram_x = act_x @ act_x.permute(0, 2, 1)
				gram_y = act_y @ act_y.permute(0, 2, 1)
				loss += torch.nn.functional.l1_loss(gram_x, gram_y)
		return loss

	def MSE_Loss(self, output1, output2, target):
		loss = self.MSELoss1(output1, output2) + self.MSELoss2(output1, target)
		return loss
	
	def forward(self, output1, output2, target):
		loss = self.MSE_Loss(output1, output2, target) + self.VGG16_Loss(output1, target)
		return loss
