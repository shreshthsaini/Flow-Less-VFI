# Importing Libraries
import numpy as np
import os, cv2
from PIL import Image
from tqdm import tqdm

import torch, timm
from torch import nn

# Save Frames to Video
def save_video(frames,video_path):
	"""
	Args:
		frames (np.array): Numpy array of frames.
		video_path (string): Video path.
	"""
	size = frames[0].shape[:2]

	video = cv2.VideoWriter(video_path,cv2.VideoWriter_fourcc(*'mp4v'), 24, (size[1], size[0]))
	for frame in frames:
		video.write(frame)
	video.release()

frames = []
for i in sorted(os.listdir("Test")):
	img = cv2.imread("Test/"+i)
	frames.append(img)

print (len(frames))
save_video(frames, 'original_videos/test_3.mp4')