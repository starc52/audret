import sys
sys.path.append('../')
import argparse
import os
import torch
import time
import torch.nn as nn
import numpy as np
from dataloader import * 
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.utils import *
from tqdm import tqdm
from scipy import spatial
from scipy.io import wavfile
import librosa
from vggm import VGGM
from resnet import *
from senet import *
from facenet_pytorch import InceptionResnetV1
from signal_utils import preprocess
import warnings
warnings.filterwarnings('ignore')

class L2Norm(nn.Module):
	def __init__(self):
		super(L2Norm, self).__init__()

	def forward(self, x):
		x = x / torch.norm(x, dim=1, keepdim=True)
		return x

class VGGFace2Model(nn.Module):
	def __init__(self, 
				embedding_dim=4096,
				arch_type='resnet'
			):
		super(VGGFace2Model, self).__init__()
		print(arch_type)
		if arch_type=='resnet':
			self.backbone = resnet50('/home/starc52/audret/resnet.pth')
		else:
			self.backbone = senet50('/home/starc52/audret/senet.pth')
		
		print("Weights Loaded!")

		for param in self.backbone.parameters():
			# param.requires_grad = True
			param.requires_grad = False

	def forward(self, x):
		batch_size = x.size(0)
		x = self.backbone(x)[1]
		x = x.view(batch_size, -1)
		# x = self.fc(x)
		
		return x


class FaceFeatureExtractor(nn.Module):
	def __init__(self, 
				embedding_dim=256
			):
		super(FaceFeatureExtractor, self).__init__()

		self.backbone = InceptionResnetV1(pretrained='vggface2')

		for param in self.backbone.parameters():
			param.requires_grad = False

		# self.fc = nn.Sequential(
		# 	nn.Linear(512, embedding_dim),
		# 	nn.ReLU(inplace=True)
		# )
		# self.activation = nn.Sigmoid()

	def forward(self, x):
		x = self.backbone(x)
		# x = self.fc(x)
		# x = x / torch.norm(x, dim=1, keepdim=True)
		# # x = self.activation(x)
		
		return x

class ImageInference():
	def __init__(self, model_val_path, arch_type="senet"):
		self.model_val_path = model_val_path
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		print(self.device)

		if arch_type=="facenet":
			self.model = FaceFeatureExtractor()
			self.model.load_state_dict(torch.load(model_val_path))
		else:
			self.model = VGGFace2Model(arch_type=arch_type)

		print("Model weights loaded from", model_val_path)
		self.model.to(self.device)
		self.model.eval()

		self.img_transform = ImageTransform(arch_type=arch_type)
		

	def get_signature(self, img_path, save_path=None, img=None):
		# img = self.img_transform.transform(img_path, img=img)
		with torch.no_grad():
			dev_img = img.unsqueeze(0)
			batch_img = dev_img.to(self.device)
			embedding = self.model(batch_img)

		embedding = embedding.cpu().numpy()
		if save_path is not None:
			np.save(save_path, embedding)
			return
		return embedding

class AudioInference():
	def __init__(self):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		print(self.device)

		self.model=VGGM(1251)
		# self.model.load_state_dict(torch.load("../VGGVox-PyTorch-master/models/VGGM300_BEST_140_81.99.pth", map_location=device))
		self.model.load_state_dict(torch.load("/home/starc52/audret/models/VGGM300_BEST_140_81.99.pth", map_location=self.device))
		self.model.to(self.device)
		print("VggVox model loaded")
		self.model.eval()
		# self.preprocess = preprocess(arch_type=arch_type)
		

	def get_audio_features(self, audio_path, save_path=None, img=None):

		audio,sr=librosa.load(audio_path,sr=16000)

		audio = audio[0:48000]

		audio = preprocess(audio).astype(np.float32)
		audio=np.expand_dims(audio, 2)
		transformers=transforms.ToTensor()
		audio = transformers(audio)
		audio = audio.unsqueeze(0)

		audio = audio.to(self.device)

		with torch.no_grad():
			audio_embedding = self.model(audio)

		audio_embedding = audio_embedding.cpu().numpy()
	
		return audio_embedding

	# todo
	#takes audio as input, splits it with window and stride params, then comptues the embeddings
	def split_audio(self, audio, window=3, stride=2, sr=16000) :
		# window and stride is in seconds

		if type(audio)==str:
			pathToAudio = audio
			audio,sr=librosa.load(audio,sr=16000)

		nsamples_per_window = int(window*sr)
		samples_per_stride = int(stride*sr)

		slice_start = [i for i in range(0,len(audio)-nsamples_per_window,samples_per_stride)]
		slice_end = [i+nsamples_per_window if i+nsamples_per_window<len(audio) else len(audio) for i in slice_start]
		
		audios = [audio[slice_start[i]:slice_end[i]] for i in range(len(slice_start))]
		audios_processed = [preprocess(audio).astype(np.float32) for audio in audios]
		audios_expanded = [np.expand_dims(audio, 2) for audio in audios_processed]
		transformers = transforms.ToTensor()
		audios_transformed = [transformers(audio) for audio in audios_expanded]
		audios_unsqzd = [audio.unsqueeze(0) for audio in audios_transformed]
		embeddings = []
		try:
			os.makedirs(pathToAudio[:-4])
			for idx, audio in enumerate(audios_unsqzd):
				audio_temp = audio.to(self.device)
				with torch.no_grad():
					audio_embedding = self.model(audio_temp)
				audio_embedding = audio_embedding.cpu().numpy()
				np.save(join(pathToAudio[:-4], "%3d.npy"%idx), audio_embedding)
				embeddings.append(audio_embedding)
		except:
			print("directory present")
		
		return embeddings

parser = argparse.ArgumentParser()
parser.add_argument('--root',default='/scratch/starc52/VoxCeleb2/dev/mp4', type=str)
parser.add_argument('--model_val_path',default="/home/starc52/Recode/senet.pth", type=str)
parser.add_argument('--start_id', default=0, type=int)
parser.add_argument('--end_id', default=5994, type=int)
args = parser.parse_args()
print(args)

ImageModel = ImageInference(args.model_val_path)
AudioModel = AudioInference()

criterion_kldiv = nn.KLDivLoss()
criterion_mse = nn.MSELoss()
criterion_cosine = nn.CosineSimilarity()

kldiv_loss = AverageMeter()
mse_loss = AverageMeter()
cosine_loss = AverageMeter()
listOfIdentities=sorted(os.listdir(args.root))[args.start_id:args.end_id]
for speaker_id in tqdm(sorted(os.listdir(args.root))[args.start_id:args.end_id]):#just 1000 user ids. 
	for url in sorted(os.listdir(join(args.root, speaker_id))):
		# os.makedirs(join(args.save_path, speaker_id, url), exist_ok=True)
		for file_name in sorted(os.listdir(join(args.root, speaker_id, url))):
			try:
				if file_name[-4:] == ".mp4":
					# print(file_name)
					cap = cv2.VideoCapture(join(args.root, speaker_id, url, file_name))
					
					ret, img_file=cap.read()
					count=0
					while cap.isOpened():
						ret, frame = cap.read()
						img_file=np.array(frame, copy=True)
						if ret and random.choice([0, 0, 0, 0, 1]):
							img_file = np.array(frame, copy=True)
							cap.release()
							break
						elif not ret:
							cap.release()
							break
						count+=1
						cap.set(1, count)
					transformToTensor = transforms.ToTensor()
					# img_file = np.expand_dims(img_file, 2)
					img_file = transformToTensor(img_file)
					face_embedding = ImageModel.get_signature(img_path="", save_path=join(args.root, speaker_id, url, file_name[:-4]+".npy"), img=img_file)
					os.remove(join(args.root, speaker_id, url, file_name))
				elif file_name[-4:] == ".wav":
					utteranceEmbedding = AudioModel.split_audio(join(args.root, speaker_id, url, file_name))
					os.remove(join(args.root, speaker_id, url, file_name))
			except:
				continue

print(listOfIdentities)
