# Importing Libraries
import numpy as np
import os, cv2, imageio
from PIL import Image
from tqdm import tqdm

import torch, timm
from torch import nn

import datasets
import models
import utils


# Plot Frames
def plot_frames(frames):
	for i in range(frames.shape[0]):
		img = frames[i].transpose(1,2,0)
		img = Image.fromarray(np.uint8(img))
		img.save("plots/"+str(i)+".png")


# Extract Frames
def extract_frames(video_path):
	"""
	Args:
		video_path (string): Video path.
	Returns:
		frames (np.array): Numpy array of frames.
	"""
	video = cv2.VideoCapture(video_path)
	success,image = video.read()

	flip = True
	frames = []
	while success:
		image = cv2.resize(image, (288,512), interpolation=cv2.INTER_AREA)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		frames.append(image)

		success,image = video.read()

	return np.array(frames).transpose(0,3,1,2)


# Save Frames to Video
def save_video(frames,video_path):
	"""
	Args:
		frames (np.array): Numpy array of frames.
		video_path (string): Video path.
	"""
	size = frames.shape[2:]
	frames = list(frames.transpose(0,2,3,1))

	video = cv2.VideoWriter(video_path,cv2.VideoWriter_fourcc(*'mp4v'), 24, (size[1], size[0]))
	for frame in frames:
		video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
	video.release()


# Predict a frame
def predict(
	video_split1,
	video_split2,
	Model
):
	"""
	Args:
		video_split1: Video split-1 (V_{1}).
		video_split2: Video split-2 (V_{2}).
		Model (torch.nn.Module): Pytorch Model.
	"""
	# Torch Tensors
	reversed_video_split2 = np.copy(video_split2[::-1,:,:,:])
	V1 = torch.Tensor(video_split1/255.0).unsqueeze(dim=0)
	V2 = torch.Tensor(reversed_video_split2/255.0).unsqueeze(dim=0)
	
	# Predictions
	with torch.no_grad():	
		y_pred1 = Model(V1.cuda()).cpu().detach().numpy()
		y_pred2 = Model(V2.cuda()).cpu().detach().numpy()
		y_pred1 = np.uint8(np.clip(y_pred1, 0, 1)*255)
		y_pred2 = np.uint8(np.clip(y_pred2, 0, 1)*255)

	return y_pred1, y_pred2


# Create a SloMo
def create_slowmo(frames, Model):
	"""
	Args:
		frames (np.array): Numpy array of video.
		Model (torch.nn.Module): Pytorch Model.
	"""
	for i in tqdm(range(0,frames.shape[0],4)):
		if i+8 < frames.shape[0]:
			V1 = frames[i:i+4]
			V2 = frames[i+4:i+8]

			y_pred1, y_pred2 = predict(V1,V2,Model)
			if i == 0:
				slomo_frames = np.concatenate((V1,y_pred1,y_pred2), axis=0)
			else:
				slomo_frames = np.concatenate((slomo_frames,V1,y_pred1,y_pred2), axis=0)
		else:
			slomo_frames = np.concatenate((slomo_frames,frames[i:]), axis=0)

	return slomo_frames

	# for i in tqdm(range(0,frames.shape[0],5)):
	# 	if i+9 < frames.shape[0]:
	# 		V1 = frames[i:i+4]
	# 		F = np.expand_dims(frames[i+4], axis=0)
	# 		V2 = frames[i+5:i+9]

	# 		y_pred1, y_pred2 = predict(V1,V2,Model)
	# 		if i == 0:
	# 			slomo_frames = np.concatenate((V1,y_pred1,F,y_pred2), axis=0)
	# 		else:
	# 			slomo_frames = np.concatenate((slomo_frames,V1,y_pred1,F,y_pred2), axis=0)
	# 	else:
	# 		slomo_frames = np.concatenate((slomo_frames,frames[i:]), axis=0)

	# return slomo_frames


def convert_video2gif(
		video_path, 
		gif_path, 
		gif_fps=24
	):
	"""
	Args:
		video_path (string): Path to video.
		gif_path (string): Path to gif.
		gif_fps (int): The fps of gif.
	"""
	video = cv2.VideoCapture(video_path)
	image_lst = []

	success,image = video.read()
	while success:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image_lst.append(image)
		success,image = video.read()

	imageio.mimsave(gif_path, image_lst, fps=gif_fps)


# Video Paths
original_videos_path = "test_samples/original_videos"
resized_videos_path = "test_samples/resized_videos"
slomo_videos_path = "test_samples/slomo_videos"
slomo_gifs_path = "test_samples/slomo_gifs"

if os.path.exists(original_videos_path) and os.path.exists(resized_videos_path) and os.path.exists(slomo_videos_path) and os.path.exists(slomo_gifs_path):
	None
else:
	assert False, "Paths provided don't exist"

# Video and GIF File
video_file = "test_3.mp4"
gif_file = "test_3.gif"

# Save Resized Video
resized_frames = extract_frames(os.path.join(original_videos_path, video_file))
save_video(resized_frames, os.path.join(resized_videos_path, video_file))

# Model
Model = models.Modified_VRT_Config0()
Model = utils.Load_Model(Model, "checkpoints/davis/modified_vrt_config0/best_model.ckpt")
Model.cuda().eval()

# Creating SlowMo video
slomo_frames = create_slowmo(resized_frames, Model)
for i in range(1):
	slomo_frames = create_slowmo(slomo_frames, Model)
print (slomo_frames.shape, resized_frames.shape)
save_video(slomo_frames, os.path.join(slomo_videos_path, video_file))

# Converting to GIF
convert_video2gif(os.path.join(slomo_videos_path, video_file), os.path.join(slomo_gifs_path, gif_file))