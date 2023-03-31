# Importing Libraries
import os

import torch, timm
from torch import nn
import torchmetrics
from torchmetrics import PeakSignalNoiseRatio as PSNR
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchinfo import summary
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.strategies.ddp import DDPStrategy

import datasets
import models
import utils
import arguments
import losses
import optimizers_schedulers


# Lightning Module
class Model_LightningModule(pl.LightningModule):
	def __init__(self, args):
		super().__init__()
		self.args = args

		# Model
		self.model = Training_Model
		self.save_hyperparameters()

		# Loss
		self.train_lossfn = losses.SelfSupervised_PerceptualConvergenceLoss()
		self.val_lossfn = losses.PerceptualConvergenceLoss()

		# Metrics
		self.psnr = PSNR()
		self.val_psnr = PSNR()
		self.val_msssim = MSSSIM(kernel_size=7)
		# self.val_lpips = LPIPS()
		

	# Training-Step
	def training_step(self, batch, batch_idx):
		x1, x2, y = batch
		batch_size = y.shape[0]

		x = torch.concat([x1,x2], dim=0)
		y_pred = self.model(x)
		y1_pred, y2_pred = y_pred[0:batch_size], y_pred[batch_size:]

		train_loss = self.train_lossfn(y1_pred, y2_pred)

		self.psnr(y1_pred, y)
		self.log('psnr', self.psnr, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

		return train_loss

	# Validation-Step
	def validation_step(self, batch, batch_idx):
		x1, x2, y = batch
		batch_size = y.shape[0]

		x = torch.concat([x1,x2], dim=0)
		y_pred = self.model(x)
		y1_pred, y2_pred = y_pred[0:batch_size], y_pred[batch_size:]

		val_loss = self.val_lossfn(y1_pred, y2_pred, y)
		self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

		self.val_psnr(y1_pred, y)
		self.val_msssim(y1_pred, y)
		
		self.log('val_psnr', self.val_psnr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
		self.log('val_msssim', self.val_msssim, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
		
	# Configure Optimizers
	def configure_optimizers(self):
		optimizer = optimizers_schedulers.optimizers(parameters=self.model.parameters(), args=self.args)
		scheduler = optimizers_schedulers.schedulers(optimizer=optimizer, args=self.args)
		if scheduler is None:
			return optimizer
		else:
			return [optimizer], [scheduler]


# Main Function
def main(args):
	# Names
	model_name = args.model_name
	dataset_name = args.dataset_name


	# Get Datasets
	Train_Dataloader = Dataset.train_dataloader()
	Valid_Dataloader = Dataset.val_dataloader()


	# Lightning Module
	Model = Model_LightningModule(args)


	# Checkpoint Callbacks
	best_checkpoint_callback = ModelCheckpoint(
		monitor="val_loss",
		save_top_k=1,
		mode="min",
		dirpath=os.path.join(args.main_path,"checkpoints",dataset_name,model_name),
		filename="self_supervised_best_model",
	)
	last_checkpoint_callback = ModelCheckpoint(
		save_last=True,
		dirpath=os.path.join(args.main_path,"checkpoints",dataset_name,model_name),
	)


	# Resume Training from checkpoint.
	if args.resume_ckpt_path is not None:
		if os.path.isfile(args.resume_ckpt_path):
			print ("Found the checkpoint at resume_ckpt_path provided.")
		else:
			args.resume_ckpt_path = None	# The given variable is altered as it is provided as input to ".fit".
			print("Resume checkpoint not found in the resume_ckpt_path provided. Starting training from the begining.")
	else:
		print ("No path is provided for resume checkpoint (resume_ckpt_path) provided. Starting training from the begining.")


	# Load Weights before the start of this particular training i.e before Epoch-0.
	if (args.load_weights_ckpt_path is not None) and (args.resume_ckpt_path is None):
		if os.path.isfile(args.load_weights_ckpt_path):
			print ("Loading Weights provided before the start of this particular training i.e before Epoch-0.")
			Model.load_from_checkpoint(args.load_weights_ckpt_path)
		else:
			print ("No checkpoint found to load weights (load_weights_ckpt_path) in the path provided.")


	# PyTorch Lightning Trainer
	trainer = pl.Trainer(
		accelerator="gpu",
		strategy=DDPStrategy(find_unused_parameters=False),
		devices = args.gpu,
		callbacks=[best_checkpoint_callback, last_checkpoint_callback, utils.LitProgressBar()],
		num_nodes=args.num_nodes,
		max_epochs=args.epochs,
		logger=pl_loggers.TensorBoardLogger(save_dir=args.main_path)
	)


	# Training the Model
	if args.train:
		print ("-"*25 + " Starting Training " + "-"*25)
		trainer.fit(Model, train_dataloaders=Train_Dataloader, val_dataloaders=Valid_Dataloader, ckpt_path=args.resume_ckpt_path)
		print ("Final Evaluation of Training Dataset")
		trainer.validate(Model, Train_Dataloader, ckpt_path=args.resume_ckpt_path)
		print ("Final Evaluation of Validation Dataset")
		trainer.validate(Model, Valid_Dataloader, ckpt_path=args.resume_ckpt_path)


	# Evaluate the Model
	if args.evaluate:
		print ("-"*25 + " Starting Evaluation " + "-"*25)
		trainer.validate(Model, Valid_Dataloader, ckpt_path=args.ckpt_path)


# Calling Main function
if __name__ == '__main__':
	root_dir = os.path.dirname(os.path.realpath(__file__))

	# Get Arguments
	args = arguments.Parse_Arguments()

	# Training Model
	Training_Model = models.Modified_VRT_Config0()
	# summary(Training_Model, input_data=torch.randn(1,3,224,224), col_names=("input_size","output_size","num_params","mult_adds"), col_width=22)

	# Dataset
	Dataset = datasets.DAVIS_Module(args)

	# Name-Arguments
	args.dataset_name = "davis"
	args.model_name = "modified_vrt_config0"

	# Main Function
	main(args)