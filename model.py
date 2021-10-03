import torch
import torch.nn as nn
from loss import *
from train import train
from senet import *
from resnet import *
from vggm import *
from facenet_pytorch import InceptionResnetV1



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
			param.requires_grad = True
			# param.requires_grad = False

	def forward(self, x):
		batch_size = x.size(0)
		x = self.backbone(x)[1]
		x = x.view(batch_size, -1)
		# x = self.fc(x)
		
		return x

class LearnablePINSenetVggVox256(nn.Module):

    def __init__(self, embedding_dim=256, training=True):

        super(LearnablePINSenetVggVox256, self).__init__()
        self.face_model = VGGFace2Model(arch_type='senet')
        self.face_model.train()
        self.face_fc = nn.Sequential(
            nn.Linear(2048, embedding_dim),
            nn.ReLU(inplace=True),
            L2Norm()
        )
        self.audio_model = VGGM(1251)
        self.audio_model.train()
        self.audio_fc = nn.Sequential(
            nn.Linear(4096, embedding_dim),
            nn.ReLU(inplace=True),
            L2Norm()
        )
        self.training=training
        self.curr_mining = CurriculumMining()

    def test(self):
        self.training=False

    def forward(self, face, audio, tau=None):
        batch_size = face.size(0)
        face = self.face_model(face)
        face = self.face_fc(face)

        audio = self.audio_model(audio)
        audio = self.audio_fc(audio)
        
        if self.training:
            positive_pairs = (face, audio)
            negative_pairs = self.curr_mining(positive_pairs, tau)
                
            return positive_pairs, negative_pairs
        else:
            return face, audio

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


class L2Norm(nn.Module):
	def __init__(self):
		super(L2Norm, self).__init__()

	def forward(self, x):
		x = x / torch.norm(x, dim=1, keepdim=True)
		return x