# Importing Libraries
from PIL import Image
import os

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder, CIFAR10
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# Video Frame Interpolation DAVIS Dataset
class VFI_DAVIS(torch.utils.data.Dataset):
	"""
	Modified from https://github.com/tarun005/FLAVR/blob/main/dataset/Davis_test.py
	"""
	def __init__(self, 
	    	data_root,
		    spatial_shape,
			num_frames=9, 
			split_min_num_frames=4,
			skip_frames=1, 
			ext="jpg"):
		"""
		Args:
			data_root (string): Path of image category folders.
			spatial_shape (tuple[int]): Dimensions of image/video spatially.
			num_frames (int): No.of frames considered for an entire video clip. Only one frame of all frames of the video will be used as target during training. (default: 11)
			skip_frames (int): No.of frames to skip between two frames of video.
			ext (string): Extension of images.
		"""
		super().__init__()
		assert num_frames >= (1 + 2*split_min_num_frames) 
		self.spatial_shape = spatial_shape
		self.num_frames = num_frames
		self.split_min_num_frames = split_min_num_frames
		

		self.images_sets = []
		for label in os.listdir(data_root):
			category_images = sorted(os.listdir(os.path.join(data_root, label)))
			category_images = [os.path.join(data_root , label , img_id) for img_id in category_images]
			for start_idx in range(0,len(category_images),skip_frames*num_frames):
				add_files = category_images[start_idx:start_idx+skip_frames*num_frames:skip_frames]
				if len(add_files) == num_frames:
					self.images_sets.append(add_files)

		self.transforms()
	

	def transforms(self):
		self.training_transforms = transforms.Compose([
				transforms.Resize((self.spatial_shape)),
				transforms.ToTensor()
		])


	def __getitem__(self, idx):
		imgpaths = self.images_sets[idx]
		images = [Image.open(img) for img in imgpaths]
		images = [self.training_transforms(img).unsqueeze(0) for img in images]
		images = torch.concat(images, dim=0)
		
		videosplit1 = images[0:int(self.num_frames/2)]
		videosplit2 = images[int(self.num_frames/2)+1:self.num_frames].flip(dims=[0])
		target = images[int(self.num_frames/2)]

		return videosplit1, videosplit2, target


	def __len__(self):
		return len(self.images_sets)
	

# DAVIS Dataset Data Loading Module
class DAVIS_Module(pl.LightningDataModule):
	def __init__(self,
		args
	) -> None:
		"""
		DAVIS dataset Module
		Args:
			args: Arguments
		"""
		super().__init__()
		self.batch_size = args.batch_size
		self.num_workers = args.num_workers

		self.dataset = VFI_DAVIS(args.data, args.image_shape)
		train_size = int(0.8 * len(self.dataset))
		test_size = len(self.dataset) - train_size
		self.train, self.val = torch.utils.data.random_split(self.dataset, [train_size, test_size])


	def train_dataloader(self):
			return DataLoader(self.train, batch_size=self.batch_size, num_workers = self.num_workers, shuffle=True)

	def val_dataloader(self):
			return DataLoader(self.val, batch_size=self.batch_size, num_workers = self.num_workers)

	def test_dataloader(self):
			return DataLoader(self.val, batch_size=self.batch_size, num_workers = self.num_workers)
