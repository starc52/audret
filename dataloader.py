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
				url_path = join(speaker_id_path, _url)
				listOfImages=[f for f in os.listdir(url_path) if os.path.isfile(join(url_path, f)) and f[-4:]==".npy"]
				for embed_path in sorted(listOfImages):
					speaker_arr = []

					if not os.path.isdir(join(root, speaker_id, _url, embed_path[:-4])) or len(os.listdir(join(root, speaker_id, _url, embed_path[:-4])))==0:
						missing+=1
						continue
					_instances = glob(join(url_path, '*/*.npy'))
					try:
						speaker_arr.append(_instances[random.randint(0, len(_instances)-1)])
					except:
						print(url_path)
						print("instances", _instances)
					_dic =  {
						'face_embed' : join(root, speaker_id, _url, embed_path),
						'speaker_embed' : speaker_arr[0]
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
		face_embed_path = self.samples_path[idx]['face_embed']
		speaker_embed_path = self.samples_path[idx]['speaker_embed']

		face_embedding = np.load(face_embed_path).reshape(-1, )
		face_embedding = face_embedding / np.linalg.norm(face_embedding)

		face_embedding = torch.from_numpy(face_embedding)

		speaker_embedding = np.load(speaker_embed_path).reshape(-1, )
		speaker_embedding = speaker_embedding / np.linalg.norm(speaker_embedding)

		speaker_embedding = torch.from_numpy(speaker_embedding)

		return face_embedding, speaker_embedding


# class ShuffledPositiveUtteranceEmbeddingLoader(Dataset):
# 	''' Add speaker centroid, generate all positive pairs and add correspondance labels '''
# 	def __init__(
# 			self,
# 			face_embed_root,
# 			speaker_embed_root,
# 			split='train'
# 		):

# 		self.face_embed_root = face_embed_root
# 		self.speaker_embed_root = speaker_embed_root
# 		self.samples_path = []

# 		missing = 0
# 		speaker_list = os.listdir(speaker_embed_root)
		
# 		# if split=='inference':
# 		# 	print("Inference on 20 speakers")
# 		# 	speaker_list = speaker_list[:20]

# 		for speaker_id in speaker_list:
# 			speaker_id_path = join(speaker_embed_root, speaker_id)
# 			speaker_arr = []
# 			face_arr = []
# 			idx = []

# 			for _url in os.listdir(speaker_id_path):
# 				url_path = join(speaker_id_path, _url)
# 				for embed_path in os.listdir(url_path):
# 					if not os.path.exists(join(face_embed_root, speaker_id, _url, embed_path)):
# 						missing+=1
# 						continue


# 					speaker_arr.append(join(url_path, embed_path))
# 					face_arr.append(join(self.face_embed_root, speaker_id, _url, embed_path))

# 			idx = np.arange(0, len(speaker_arr))

# 			neg_flag = 0
# 			diff = np.arange(0, len(speaker_arr))
# 			while np.sum(np.where(diff==0)) > 0:
# 			    random.shuffle(idx)
# 			    true_indices = np.arange(0,len(speaker_arr))
# 			    diff = true_indices - idx


# 			for _idx, face in enumerate(face_arr):
# 				_dic =  {
# 					'face_embedding' : face,
# 					'speaker_embedding' : speaker_arr[idx[_idx]],
# 					'_id' : 'dummy path' 
# 				}
# 				self.samples_path.append(_dic)


# 		print("Missing %d, Total count %d, Split: %s"%(missing, len(self.samples_path), split))

# 		if(split=='train'):
# 		    random.shuffle(self.samples_path)

# 	def __len__(self):
# 		return len(self.samples_path)

# 	def __getitem__(self, idx):
# 		''' 
# 			L2 Normalizing both embeddings
# 		'''
# 		face_embedding_path = self.samples_path[idx]['face_embedding']
# 		speaker_embedding_path = self.samples_path[idx]['speaker_embedding']
# 		_id = self.samples_path[idx]['_id']

# 		face_embedding = np.load(face_embedding_path).reshape(-1, )
# 		speaker_embedding = np.load(speaker_embedding_path)

# 		face_embedding = face_embedding / np.linalg.norm(face_embedding)
# 		speaker_embedding = speaker_embedding / np.linalg.norm(speaker_embedding)

# 		face_embedding = torch.from_numpy(face_embedding)
# 		speaker_embedding = torch.from_numpy(speaker_embedding)

# 		return face_embedding, speaker_embedding, _id



# class UtteranceEmbeddingLoader(Dataset):
# 	''' Add speaker centroid, generate all positive pairs and add correspondance labels '''
# 	def __init__(
# 			self,
# 			face_embed_root,
# 			speaker_embed_root,
# 			split='train'
# 		):

# 		self.face_embed_root = face_embed_root
# 		self.speaker_embed_root = speaker_embed_root
# 		self.samples_path = []

# 		missing = 0
# 		speaker_list = os.listdir(speaker_embed_root)
# 		if split=='inference':
# 			print("Inference on 20 speakers")
# 			speaker_list = speaker_list[:20]

# 		for speaker_id in speaker_list:
# 			speaker_id_path = join(speaker_embed_root, speaker_id)
# 			for _url in os.listdir(speaker_id_path):
# 				url_path = join(speaker_id_path, _url)
# 				for embed_path in os.listdir(url_path):
# 					if not os.path.exists(join(face_embed_root, speaker_id, _url, embed_path)):
# 						missing+=1
# 						continue

# 					_dic =  {
# 						'face_embedding' : join(self.face_embed_root, speaker_id, _url, embed_path),
# 						'speaker_embedding' : join(url_path, embed_path),
# 						'_id' : embed_path.split('.npy')[0] 
# 					}
# 					self.samples_path.append(_dic)

# 		print("Missing %d, Total count %d, Split: %s"%(missing, len(self.samples_path), split))

# 		if(split=='train'):
# 		    random.shuffle(self.samples_path)

# 	def __len__(self):
# 		return len(self.samples_path)

# 	def __getitem__(self, idx):
# 		''' 
# 			L2 Normalizing both embeddings
# 		'''
# 		face_embedding_path = self.samples_path[idx]['face_embedding']
# 		speaker_embedding_path = self.samples_path[idx]['speaker_embedding']
# 		_id = self.samples_path[idx]['_id']

# 		face_embedding = np.load(face_embedding_path).reshape(-1, )
# 		speaker_embedding = np.load(speaker_embedding_path)

# 		face_embedding = face_embedding / np.linalg.norm(face_embedding)
# 		speaker_embedding = speaker_embedding / np.linalg.norm(speaker_embedding)

# 		face_embedding = torch.from_numpy(face_embedding)
# 		speaker_embedding = torch.from_numpy(speaker_embedding)

# 		return face_embedding, speaker_embedding, _id


# # a = UtteranceEmbeddigLoader('/ssd_scratch/cvit/samyak/voxceleb_senet_face_embeddings/train/', '/ssd_scratch/cvit/samyak/vgg_vox_embeddings/train_vgg_vox/')
# # for idx, sample in enumerate(a):
# # 	if idx<10:continue
# # 	else: break
# # 	# print(sample)

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
		#     random.shuffle(self.samples_path)
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
