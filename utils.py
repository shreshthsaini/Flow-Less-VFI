# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import os, json

import torch, torchvision
from pytorch_lightning.callbacks import TQDMProgressBar
import pytorch_lightning as pl

# Progress bar
class LitProgressBar(TQDMProgressBar):
	def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
		print ()
		return super().on_validation_epoch_end(trainer, pl_module)
	
def Plot_Images(
	Dataset,
	save_path="Plots"
):
	"""
	Plots video splits as images from dataset.
	Args:
		Dataset (torch.utils.data.Dataset): Torch Dataset.
	"""
	idx = np.random.randint(Dataset.__len__())
	V1, V2, I = Dataset.__getitem__(idx)
	plt.figure(figsize=(12, 12))
	for i in range(4):
		plt.imshow(V1[i].permute(1, 2, 0))
		plt.axis("off")
		plt.savefig(save_path+str(i)+".png")
		
	plt.imshow(I[0].permute(1, 2, 0))
	plt.savefig(save_path+str(4)+".png")
		
	for i in range(4):
		plt.imshow(V2[i].permute(1, 2, 0))
		plt.axis("off")
		plt.savefig(save_path+str(5+i)+".png")