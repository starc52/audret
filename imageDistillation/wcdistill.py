import numpy as np
import os,sys
sys.path.append('../')
import glob
import  random
# import scipy
# from scipy import spatial
import torch
import torch.nn as nn
from datetime import datetime
# from time import time 
from tqdm import tqdm
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from model import *
from PIL import Image
import cv2
import torchvision
from torchvision import transforms
from time import strftime, time
from facenet_pytorch import InceptionResnetV1
random.seed(0)
from senet import *
from resnet import * 
# from torch.utils.data import Dataset
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
import tensorboardX
from os.path import join
import pdb#; pdb.set_trace()
random.seed(0)

EMBEDDING_DIM = 2048 #512 for Facenet, 2048 for VGGface2
BATCHSIZE = 248
NUM_WORKERS = 8
ALPHA=0.6
EPOCH_NUM=10

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
                url_path = join(speaker_id_path, _url)
                listOfImages=[join(url_path, 'frames', f) for f in os.listdir(join(url_path, 'frames')) if os.path.isfile(join(url_path, 'frames', f)) and f[-4:]==".jpg"]
                self.samples_path+=listOfImages

        print("Total count %d, Split: %s"%(len(self.samples_path), split))

        if(split=='train'):
            random.shuffle(self.samples_path)


    def __len__(self):
        return len(self.samples_path)

    def __getitem__(self, idx):
        transformers=transforms.ToTensor()
        face_frame_path = self.samples_path[idx]
        
        face_frame_teacher = Image.open(face_frame_path).convert('RGB')
        face_frame_student = Image.open(face_frame_path).convert('RGB')
        
        teacher_transform = transforms.Compose([
            transforms.Resize(224),  # smaller side resized
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, hue=0.3),
            transforms.ToTensor(),
        ])
        student_transform = transforms.Compose([
            transforms.Resize(224),  # smaller side resized
            transforms.Resize(56),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, hue=0.3),
            transforms.ToTensor(),
        ])
        face_frame_teacher = teacher_transform(face_frame_teacher)
        face_frame_student = student_transform(face_frame_student)
        
        return face_frame_teacher, face_frame_student

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
            param.requires_grad = True
            # param.requires_grad = False

    def forward(self, x):
        batch_size = x.size(0)
        x = self.backbone(x)[1]
        x = x.view(batch_size, -1)
        # x = self.fc(x)
        
        return x


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

criterion = nn.MSELoss()

aud_vis_model_path=join('/ssd_scratch/cvit/starc52/LPscheckpoints','model_e49.pth')
aud_vis_model = LearnablePINSenetVggVox256()
aud_vis_model.test()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
aud_vis_model.to(device)
if torch.cuda.device_count() > 1:
    aud_vis_model = nn.DataParallel(aud_vis_model)
aud_vis_model.load_state_dict(torch.load(aud_vis_model_path)['model_state_dict'])


model = CompleteDistillationModel(aud_vis_model)

root = "/ssd_scratch/cvit/starc52/VoxCeleb2/dev/mp4/"
LOG_PATH = '/home/starc52/LearnablePINs/train_log_'+str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+"/"
CHECKPOINT_PATH = "/ssd_scratch/cvit/starc52/distill_checkpoints/"

if os.path.isdir(CHECKPOINT_PATH):
    pass
else:
    os.makedirs(CHECKPOINT_PATH)


TBoard = tensorboardX.SummaryWriter(log_dir=LOG_PATH)

train = simpleDataLoader(root)
trainloader = torch.utils.data.DataLoader(train, batch_size=BATCHSIZE, num_workers=NUM_WORKERS, shuffle=True)


optimizer = torch.optim.Adam(model.parameters())

model = model.to(device)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

for epoch_num in tqdm(range(EPOCH_NUM)):
    for iter_num, (teacher_images, student_images) in enumerate(trainloader):
        step_num = epoch_num * len(trainloader) + iter_num
        teacher_images, student_images = teacher_images.to(device, non_blocking=True), student_images.to(device, non_blocking=True)

        optimizer.zero_grad()

        emb_teacher, emb_student = model(teacher_images, student_images)

        loss = criterion(emb_teacher, emb_student)

        loss.backward()
        optimizer.step()
        
        
        TBoard.add_scalar('Train/Loss', loss.item(), step_num)
#         TBoard.add_histogram('teacher_images', emb_teacher, step_num)
#         TBoard.add_histogram('student_images', emb_student, step_num)

    torch.save({
        'epoch':epoch_num,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'loss':loss,
    }, os.path.join(CHECKPOINT_PATH, "epoch_"+str(epoch_num)+".pth"))
    teacher_img = np.squeeze(np.reshape(teacher_images[0].cpu(), (-1, 3, 224, 224)))
    student_img = np.squeeze(np.reshape(student_images[0].cpu(), (-1, 3, 224, 224)))
    TBoard.add_image('teacher_images', teacher_img, step_num)
    TBoard.add_image('student_images', student_img, step_num)


