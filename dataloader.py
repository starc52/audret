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
		# if split=='inference':
		# 	print("Inference on 20 speakers")
		# 	speaker_list = speaker_list[:20]
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
			L2 Normalizing both embeddings
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
		# face_embedding = face_embedding / np.linalg.norm(face_embedding)

		# face_embedding = torch.from_numpy(face_embedding)
		try:
			#print(speaker_fft_path)
			speaker_fft = np.load(speaker_fft_path)
			#print(speaker_fft)
		except:
			print(traceback.format_exc())
			sys.exit()
		speaker_fft = transformers(speaker_fft)
		# speaker_fft = speaker_fft.unsqueeze(0) # basically this adds batchsize as a dimension
		# speaker_embedding = speaker_embedding / np.linalg.norm(speaker_embedding)

		# speaker_embedding = torch.from_numpy(speaker_embedding)

		return face_frame, speaker_fft

class simpleDataLoaderLossy(Dataset):

	def __init__(self, loss_factor, root, split='train'):
		self.root = root
		self.loss_factor = loss_factor
		missing = 0
		speaker_list = sorted(os.listdir(root)) #/scratch/dev/VoxCeleb2/dev/mp4
		self.samples_path=[]
		# if split=='inference':
		# 	print("Inference on 20 speakers")
		# 	speaker_list = speaker_list[:20]
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
			L2 Normalizing both embeddings
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
		# face_embedding = face_embedding / np.linalg.norm(face_embedding)

		# face_embedding = torch.from_numpy(face_embedding)
		try:
			#print(speaker_fft_path)
			speaker_fft = np.load(speaker_fft_path)
			#print(speaker_fft)
		except:
			print(traceback.format_exc())
			sys.exit()
		speaker_fft = transformers(speaker_fft)
		# speaker_fft = speaker_fft.unsqueeze(0) # basically this adds batchsize as a dimension
		# speaker_embedding = speaker_embedding / np.linalg.norm(speaker_embedding)

		# speaker_embedding = torch.from_numpy(speaker_embedding)

		return face_frame, speaker_fft


face_dimension = 2048
speaker_dimension = 4096

class Embeddings(Dataset):

	'''
		KLDiv, MSE, L1 and GE2E loss can be used
	'''

	def __init__(self, 
				face_embed_root, 
				speaker_embed_root, 
				split='train'):
		
		self.face_embed_root = face_embed_root
		self.speaker_embed_root = speaker_embed_root
		self.samples_path = []

		files = os.listdir(self.face_embed_root)

		_count = 0
		for idx, f in enumerate(files):
			_dic =  {
				'face_embedding' : join(self.face_embed_root,f),
				'speaker_embedding' : join(self.speaker_embed_root,f),
				'_id' : f.split('.npy')[0] 
			}
			self.samples_path.append(_dic)
			_count += 1

		print(_count, len(self.samples_path), split)
		# if(split=='train'):
		#	 random.shuffle(self.samples_path)
		self.split = split

	def __len__(self):
		return len(self.samples_path)

	def __getitem__(self, idx):
		face_embedding_path = self.samples_path[idx]['face_embedding']
		speaker_embedding_path = self.samples_path[idx]['speaker_embedding']
		_id = self.samples_path[idx]['_id']

		face_embedding = np.load(face_embedding_path)
		face_embedding = np.reshape(face_embedding,(-1,face_dimension))

		while(face_embedding.shape[0]<10):
			face_embedding = np.vstack((face_embedding,face_embedding[-1].reshape(1,-1)))
		face_embedding = torch.from_numpy(face_embedding)
		
		assert face_embedding.shape[0]==10, 'Face embedding wrong dimension Id : %s, [%s]'%(_id,str(face_embedding.shape))
		assert face_embedding.shape[1]==face_dimension, 'Face embedding wrong dimension Id : %s, [%s]'%(_id,str(face_embedding.shape))

		speaker_embedding = np.load(speaker_embedding_path).reshape(-1,)

		if speaker_embedding.max() > 1:
			speaker_embedding = speaker_embedding / np.linalg.norm(speaker_embedding)

		speaker_embedding = np.reshape(speaker_embedding,(1,speaker_dimension))
		speaker_embedding = np.tile(speaker_embedding,(face_embedding.shape[0],1))
		assert speaker_embedding.shape[0] == face_embedding.shape[0] and speaker_embedding.shape[1]==speaker_dimension , 'Speaker embedding wrong dimension Id : %s, [%s]'%(_id,str(speaker_embedding.shape)) 
		speaker_embedding = torch.from_numpy(speaker_embedding)

		return face_embedding, speaker_embedding

random.seed(0)
np.random.seed(0)


class FaceFeatureExtractorDataloader(Dataset):
	def __init__(self, root, embedding_root_path):
		self.root = root
		self.samples_path = []

		_label = 0
		for utterance in sorted(os.listdir(embedding_root_path)):
			for links in sorted(os.listdir(join(embedding_root_path, utterance))):
				for _id in sorted(os.listdir(join(embedding_root_path, utterance, links))):
					id_num = _id.split('.')[0]
					_dic = {
						'embedding_path': join(embedding_root_path, utterance, links, _id), 
						'frame_path': join(root, utterance, links, 'frames', id_num+'.png'),
						'label': _label
					}
					self.samples_path.append(_dic)

			_label += 1

		print(_label, len(self.samples_path))
		random.shuffle(self.samples_path)
		self.img_transform = transforms.Compose([
			transforms.Resize((160, 160)),
			transforms.ToTensor(),
			fixed_image_standardization
		])

	def __len__(self):
		return len(self.samples_path)

	def __getitem__(self, idx):
		frame_path = self.samples_path[idx]['frame_path']
		embedding_path = self.samples_path[idx]['embedding_path']
		label = self.samples_path[idx]['label']

		img = Image.open(frame_path).convert('RGB')
		embedding = np.load(embedding_path)
		assert embedding.min()>=0 and embedding.max()<=1, (embedding.min(), embedding.max(), embedding_path)
		embedding = torch.from_numpy(embedding)

		img = self.img_transform(img)

		return {
			'image': img,
			'embedding': embedding,
			'label': label
		}

class VGGFace2Loader(Dataset):
	mean_bgr = np.array([91.4953, 103.8827, 131.0912])

	def __init__(self, root, embedding_root_path, split='train'):
		self.root = root
		self.samples_path = []

		_label = 0
		for utterance in sorted(os.listdir(embedding_root_path)):

			# ge2e speaker embeddings
			# embedding_path = join('..\\speaker_embeddings\\{}_centroid'.format(split), utterance+'.npy')

			# vggvox speaker embeddings
			embedding_path = join('..\\vgg_vox_speaker_embeddings\\{}_centroid_vgg_vox'.format(split), utterance+'.npy')

			for links in sorted(os.listdir(join(embedding_root_path, utterance))):
				for _id in sorted(os.listdir(join(embedding_root_path, utterance, links))):
					id_num = _id.split('.')[0]
					if not os.path.exists(join(root, utterance, links, 'frames', id_num+'.png')):
						continue
					_dic = {
						'embedding_path': embedding_path, 
						'frame_path': join(root, utterance, links, 'frames', id_num+'.png'),
						'label': _label
					}
					self.samples_path.append(_dic)

			_label += 1

		print(_label, len(self.samples_path), split)
		random.shuffle(self.samples_path)
		self.split = split

	def __len__(self):
		return len(self.samples_path)

	def transform(self, img):
		img = img[:, :, ::-1]  # RGB -> BGR
		img = img.astype(np.float32)
		img -= self.mean_bgr
		img = img.transpose(2, 0, 1)  # C x H x W
		img = torch.from_numpy(img).float()
		return img

	def __getitem__(self, idx):
		frame_path = self.samples_path[idx]['frame_path']
		embedding_path = self.samples_path[idx]['embedding_path']
		label = self.samples_path[idx]['label']

		img = Image.open(frame_path).convert('RGB')
		embedding = np.load(embedding_path)
		if not (embedding.min()>=0 and embedding.max()<=1):
			embedding = embedding / np.linalg.norm(embedding, 2)
		assert embedding.min()>=0 and embedding.max()<=1, (embedding.min(), embedding.max(), embedding_path)
		embedding = torch.from_numpy(embedding)

		img = transforms.Resize(256)(img)

		if self.split=='train':
		# 	# include random horizontalflips and random colour jitter 
			img = transforms.RandomCrop(224)(img)
			img = transforms.RandomGrayscale(p=0.2)(img)
		else:
			img = transforms.CenterCrop(224)(img)

		img = np.array(img, dtype=np.uint8)
		assert len(img.shape) == 3
		img = self.transform(img)

		return {
			'image': img,
			'image_path' : frame_path,
			'embedding': embedding,
			'label': label
		}
