from __future__ import print_function, division

import os
import pandas as pd
import numpy as np
import glob
from skimage import io, transform
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataset import Dataset  # For custom data-sets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from matplotlib.image import imread
from transforms import (
    Resize,
    Normalize,
)


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode



class ISIC2018Dataset(Dataset):
	"""ISIC 2018 Dataset."""
	def __init__(
		self, 
		root_dir, 
		img_folder,
		img_filename_format,
		msk_folder=None,
		msk_filename_format=None,
		transform_list=[], 
		*args, **kwargs):
		"""Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.root_dir = root_dir if not '/' == root_dir[-1] else os.path.dirname(root_dir)
		self.img_fp_list = glob.glob(f"{os.path.join(self.root_dir, img_folder)}/{img_filename_format}")
		self.fid_list = [fn.split('/')[-1].split('.')[0].split('_')[1] for fn in self.img_fp_list]
		
		self.img_folder = img_folder
		self.img_filename_format = img_filename_format
		self.msk_folder = msk_folder
		self.msk_filename_format = msk_filename_format

		if transform_list:
			self.transform = transforms.Compose([eval(t) for t in transform_list])
		else:
			self.transform = None
		

	def __len__(self):
		return len(self.fid_list)


	def __getitem__(self, indeimg):
		file_id = self.fid_list[indeimg]

		img_filepath = f"{os.path.join(self.root_dir, self.img_folder)}/{self.img_filename_format.replace('*', file_id)}"
		img = imread(img_filepath)
		img = torch.tensor(img).permute(2,0,1)
		
		if self.msk_folder:
			msk_filepath = f"{os.path.join(self.root_dir, self.msk_folder)}/{self.msk_filename_format.replace('*', file_id)}"
			msk = imread(msk_filepath)
			if len(msk.shape) < 3: 
				msk = torch.tensor(msk).unsqueeze(-1).permute(2,0,1)
			else: 
				msk = torch.tensor(msk).permute(2,0,1)
			sample = {'img': img, 'msk': msk}
		else:
			sample = {'img': img}

		if self.transform:
			sample = self.transform(sample)

		return sample
