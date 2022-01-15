import torch
import torch.nn as nn
from loss import *
from train import train
from senet import *
from resnet import *
from vggm import *
from facenet_pytorch import InceptionResnetV1

EMBEDDING_DIM =2048

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

class LearnablePINSdistill(nn.Module):

    def __init__(self, embedding_dim=256, training=True):

        super(LearnablePINSdistill, self).__init__()
        self.face_model = SmallDistilledModel(embedding_dim = EMBEDDING_DIM)
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

def conv_layer(channel_in, channel_out, k_size, p_size, stride, dil):
    layer = nn.Sequential(
        nn.Conv2d(channel_in, channel_out, kernel_size=k_size, padding=p_size, stride=stride, dilation=dil),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(inplace=True)
    )
    return layer

def linear_layer(size_in, size_out, l2=False):
    if l2:
        layer = nn.Sequential(
            nn.Linear(size_in, size_out),
            nn.ReLU(inplace=True),
            L2Norm()
        )
        return layer
    else:
        layer = nn.Sequential(
            nn.Linear(size_in, size_out),
            nn.ReLU(inplace=True),
        )
        return layer

class CompleteDistillationModel(nn.Module):

    def __init__(self, aud_vis_model):
        super(CompleteDistillationModel, self).__init__()
        self.training=True
        self.teacher_model = VGGFace2Model(arch_type=2048)
        pretrained_dict = aud_vis_model.module.face_model.state_dict()
        vgg_model_dict = self.teacher_model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in vgg_model_dict}
        # 2. overwrite entries in the existing state dict
        vgg_model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.teacher_model.load_state_dict(vgg_model_dict)
        self.student_model = SmallDistilledModel(embedding_dim=EMBEDDING_DIM)
        for param in self.teacher_model.parameters():
            param.requires_grad=False
        
    def forward(self, teacher_images, student_images, tau=None):
        emb_teachers = self.teacher_model(teacher_images)
        emb_students = self.student_model(student_images)
        return emb_teachers, emb_students

class SmallDistilledModel(nn.Module):

    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super(SmallDistilledModel, self).__init__()
        self.layer1 = conv_layer(3, 8, 4, 0, 2, 2) #Output:8, 109, 109
        self.layer2 = conv_layer(8, 16, 4, 0, 2, 2) #Output:16, 52, 52
        self.maxpool1 = nn.MaxPool2d(2) #Output: 16, 26, 26
        self.layer3 = conv_layer(16, 32, 3, 1, 2, 1) #Output:32, 13, 13
        self.maxpool2 = nn.MaxPool2d(2) #Output: 32, 6, 6
        self.layer4 = conv_layer(32, 16, 1, 0, 1, 1) #Output:16, 6, 6
        self.layer5 = conv_layer(16, 128, 3, 1, 1, 1) #Output:128, 6, 6
        self.maxpool3 = nn.MaxPool2d(2) #Output: 128, 3, 3
        self.layer6 = conv_layer(128, 32, 1, 0, 1, 1) #Output:32, 3, 3
        self.layer7 = conv_layer(32, 256, 3, 1, 1, 1) #Output:256, 3, 3
        self.maxpool4 = nn.MaxPool2d(2) #Output: 256, 1, 1
        self.layer8 = conv_layer(256, 256, 1, 0, 1, 1) #Output:128, 1, 1
        self.layer9 = conv_layer(256, EMBEDDING_DIM, 1, 0, 1, 1) #Output:embedding_dim, 1, 1

        # self.layer3 = conv_layer(16, 64, 4, 0, 2, 1)
        # self.layer4 = conv_layer(64, 128, 4, 0, 2, 1)
        # self.flatten = nn.Flatten()
#         self.linear1 = linear_layer(EMBEDDING_DIM, EMBEDDING_DIM, l2=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.maxpool1(x)
        
        x = self.layer3(x)
        x = self.maxpool2(x)
        
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.maxpool3(x)
        
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.maxpool4(x)
        
        x = self.layer8(x)
        x = self.layer9(x)
        
        x = x.view(x.size(0), -1)
#         x = self.linear1(x)
        
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
