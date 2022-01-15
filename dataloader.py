from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from os.path import join
import torch
import random
from torchvision import transforms
from glob import glob
from facenet_pytorch import fixed_image_standardization
from PIL import Image
import traceback
import sys

random.seed(0)


class simpleDataLoader(Dataset):

	def __init__(self, root, split='train'):
		self.root = root

		missing = 0
		speaker_list = sorted(os.listdir(root)) #/scratch/dev/VoxCeleb2/dev/mp4
		self.samples_path=[]

		print("Starting dataloading")
		for i, speaker_id in enumerate(speaker_list):
			speaker_id_path = join(root, speaker_id)

			for _url in sorted(os.listdir(speaker_id_path)):
				im_url_path = join(speaker_id_path, _url, "frames")
				au_url_path = join(speaker_id_path, _url, "audio")
				listOfImages=[f for f in os.listdir(im_url_path) if os.path.isfile(join(im_url_path, f)) and f[-4:]==".jpg"]
				listOfAudios=[f for f in os.listdir(au_url_path) if os.path.isfile(join(au_url_path, f)) and f[-4:]==".npy"]
				if len(listOfImages)==0 or len(listOfAudios)==0:
					missing+=1
					continue
				for embed_path in sorted(listOfImages):
					speaker_arr = []
					try:
						speaker_arr.append(listOfAudios[random.randint(0, len(listOfAudios)-1)])
					except:
						print(au_url_path)
						print("instances", listOfAudios)
					_dic =  {
						'face_frame' : join(root, speaker_id, _url, "frames", embed_path),
						'speaker_fft' : join(root, speaker_id, _url, "audio", speaker_arr[0])
					}
					self.samples_path.append(_dic)

		print("Missing %d, Total count %d, Split: %s"%(missing, len(self.samples_path), split))

		if(split=='train'):
			random.shuffle(self.samples_path)

	def __len__(self):
		return len(self.samples_path)

	def __getitem__(self, idx):
		''' 
			Loading face crops after transforms and audio FFTs
		'''
		transformers=transforms.ToTensor()
		face_frame_path = self.samples_path[idx]['face_frame']
		speaker_fft_path = self.samples_path[idx]['speaker_fft']

		face_frame = Image.open(face_frame_path).convert('RGB')
		img_transform = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.ColorJitter(brightness=0.5, hue=0.3),
			transforms.ToTensor(),
		])
		face_frame = transformers(face_frame)
		try:
			speaker_fft = np.load(speaker_fft_path)
		except:
			print(traceback.format_exc())
			sys.exit()
		speaker_fft = transformers(speaker_fft)
		return face_frame, speaker_fft

class simpleDataLoaderLossy(Dataset):

	def __init__(self, loss_factor, root, split='train'):
		self.root = root
		self.loss_factor = loss_factor
		missing = 0
		speaker_list = sorted(os.listdir(root)) #/scratch/dev/VoxCeleb2/dev/mp4
		self.samples_path=[]
		print("Starting dataloading")
		for i, speaker_id in enumerate(speaker_list):
			speaker_id_path = join(root, speaker_id)

			for _url in sorted(os.listdir(speaker_id_path)):
				im_url_path = join(speaker_id_path, _url, "frames")
				au_url_path = join(speaker_id_path, _url, "audio")
				listOfImages=[f for f in os.listdir(im_url_path) if os.path.isfile(join(im_url_path, f)) and f[-4:]==".jpg"]
				listOfAudios=[f for f in os.listdir(au_url_path) if os.path.isfile(join(au_url_path, f)) and f[-4:]==".npy"]
				if listOfImages==None or listOfAudios==None:
					missing+=1
					continue
				for embed_path in sorted(listOfImages):
					speaker_arr = []
					try:
						speaker_arr.append(listOfAudios[random.randint(0, len(listOfAudios)-1)])
					except:
						print(au_url_path)
						print("instances", listOfAudios)
					_dic =  {
						'face_frame' : join(root, speaker_id, _url, "frames", embed_path),
						'speaker_fft' : join(root, speaker_id, _url, "audio", speaker_arr[0])
					}
					self.samples_path.append(_dic)

		print("Missing %d, Total count %d, Split: %s"%(missing, len(self.samples_path), split))

		if(split=='train'):
			random.shuffle(self.samples_path)


	def __len__(self):
		return len(self.samples_path)

	def __getitem__(self, idx):
		''' 
			Loading face crops after transforms (with downscale and upscale) and audio FFTs
		'''
		transformers=transforms.ToTensor()
		face_frame_path = self.samples_path[idx]['face_frame']
		speaker_fft_path = self.samples_path[idx]['speaker_fft']

		face_frame = Image.open(face_frame_path).convert('RGB')
		img_transform = transforms.Compose([
			transforms.Resize((int(224*self.loss_factor), int(224*self.loss_factor))),
			transforms.Resize((224, 224)),
			transforms.RandomHorizontalFlip(),
			transforms.ColorJitter(brightness=0.5, hue=0.3),
			transforms.ToTensor(),
		])
		face_frame = img_transform(face_frame)
		try:
			speaker_fft = np.load(speaker_fft_path)
		except:
			print(traceback.format_exc())
			sys.exit()
		speaker_fft = transformers(speaker_fft)
		return face_frame, speaker_fft

