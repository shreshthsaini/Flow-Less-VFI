# Importing Libraries
import numpy as np
import os
from PIL import Image

import torch, timm
from torch import nn

import datasets
import models
import utils

def predict(
	Dataset,
	num_samples,
	ckpt_path,
	path = "results"
):
	i = 0
	samples = []
	for batch in Valid_Dataloader:
		if np.random.rand(1)[0] >= 0.5:
			samples.append(batch)
			i += 1
		else:
			None
		if i==num_samples:
			break

	# Model
	Model = models.Modified_VRT_Config0()
	Model = utils.Load_Model(Model, ckpt_path)
	Model.eval().cuda()

	for i in range(num_samples):
		# Getting a sample
		V1, V2, y = samples[i]

		with torch.no_grad():
			# Predictions
			y_pred1 = Model(V1.cuda())[0].cpu().detach().numpy()
			y_pred2 = Model(V2.cuda())[0].cpu().detach().numpy()
			y = y[0].numpy()

		# Saving Predictions
		for img,filename in zip((y_pred1, y_pred2, y, V1[0,-1].cpu().detach().numpy(), V2[0,-1].cpu().detach().numpy()),("y_pred1.png", "y_pred2.png", "y.png", "prev.png", "next.png")):
			img = np.clip(np.transpose(img, (1,2,0)), 0, 1)
			img = Image.fromarray(np.uint8(img*255))
			img.save(os.path.join(path,"sample_"+ str(i) + "_" + filename))

# Dataset
Dataset = datasets.DAVIS_Module(
	batch_size=1,
	num_workers=16,
	data="DAVIS/JPEGImages/480p",
	image_shape=(240,240)
)
Valid_Dataloader = Dataset.val_dataloader()

predict(Valid_Dataloader, 20, "checkpoints/davis/modified_vrt_config0/best_model.ckpt")