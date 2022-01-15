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
# from face2speaker_main import Face2Speaker
# from Recode.model import DoubleLayerModel, SingleLayerModel
from senet50 import *
from PIL import Image
import cv2
import torchvision
from torchvision import transforms
from time import strftime, time
from facenet_pytorch import InceptionResnetV1
random.seed(0)
from senet50 import Senet50_ft_dag
from resnet50 import Resnet50_ft_dag 
# from torch.utils.data import Dataset
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
import tensorboardX
import resnet50
import pdb#; pdb.set_trace()
random.seed(0)


EMBEDDING_DIM = 2048 #512 for Facenet, 2048 for VGGface2
BATCHSIZE = 64
RGB_MEAN = [255*i for i in [0.6071, 0.4609, 0.3944]]
RGB_STD = [255*i for i in [0.2457, 0.2175, 0.2129]]
NUM_WORKERS = 8
ALPHA=0.6
EPOCH_NUM=2


teacher_transform = transforms.Compose([
        transforms.Resize(224),  # smaller side resized
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN, std=RGB_STD), # only when using vggface
    ])
student_transform = transforms.Compose([
        transforms.Resize(224),  # smaller side resized
        transforms.RandomCrop(224),
        transforms.Resize(56),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN, std=RGB_STD), # only when using vggface
    ])

class VoxCeleb2VGGface2Dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, csv_file, teacher_transform=None, student_transform=None):
        self.img_dir = img_dir
        self.image_list = pd.read_csv(csv_file, names=['path'], index_col=False).reset_index(drop=True) # ./train/n000012/0000_00.jpg
        self.teacher_transform=teacher_transform
        self.student_transform=student_transform
    
    def __len__(self):
        return len(self.image_list.index)

    def __getitem__(self, idx):
        teach_image_path = os.path.join(self.img_dir, self.image_list.loc[idx, 'path'][2:])
        teach_image = Image.open(teach_image_path).convert('RGB')
        image_iden = "/".join(teach_image_path.split("/")[:-1])
        iden_list = random.sample(os.listdir(image_iden), 10)
        temp_iden_list = [os.path.join(image_iden, i) for i in iden_list]
        iden_list = temp_iden_list
        iden_list.append(teach_image_path)
        student_image = Image.open(random.choice(iden_list)).convert('RGB')
        teach_image = self.teacher_transform(teach_image)
        student_image = self.student_transform(student_image)
        return teach_image, student_image


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()

    def forward(self, x):
        x = x / torch.norm(x, dim=1, keepdim=True)
        return x


class FaceNetVGGFaceModels(nn.Module):
    def __init__(self, teach_model = 'vggface2', arch_type='senet'):
        super(FaceNetVGGFaceModels, self).__init__()
        if teach_model=='vggface2':
            if arch_type == 'senet':
                self.vggface2 = Senet50_ft_dag()
                self.vggface2.load_state_dict(torch.load("/home/starc52/verandret/senet50_ft_dag.pth"))
            else:
                self.vggface2 = Resnet50_ft_dag()
                self.vggface2.load_state_dict(torch.load("/home/starc52/verandret/resnet50_ft_dag.pth"))
            for param in self.vggface2.parameters():
                param.requires_grad=False
        else:
            self.facenetInception = InceptionResnetV1(pretrained='vggface2')
            for param in self.facenetInception.parameters():
                param.requires_grad=False

        if teach_model == 'vggface2':
            self.backbone = self.vggface2
        else:
            self.backbone = self.facenetInception

    def forward(self, x):
        x = self.backbone(x)[1] #[1] is vggface2 specific
        # x = self.fc(x.view(x.size(0), -1))
        x = x.view(x.size(0), -1)
        return torch.nn.functional.normalize(x)

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
        self.layer9 = conv_layer(256, EMBEDDING_DIM//2, 1, 0, 1, 1) #Output:embedding_dim, 1, 1

        # self.layer3 = conv_layer(16, 64, 4, 0, 2, 1)
        # self.layer4 = conv_layer(64, 128, 4, 0, 2, 1)
        # self.flatten = nn.Flatten()
        self.linear1 = linear_layer(EMBEDDING_DIM//2, EMBEDDING_DIM, l2=True)

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
        x = self.linear1(x)
        
        return x

class CompleteDistillationModel(nn.Module):

    def __init__(self):
        super(CompleteDistillationModel, self).__init__()
        self.training=True
        self.teacher_model = FaceNetVGGFaceModels(teach_model="vggface2", arch_type="senet")
        self.student_model = SmallDistilledModel(embedding_dim=EMBEDDING_DIM)
        for param in self.teacher_model.parameters():
            param.requires_grad=False
        self.curr_mining = CurriculumMining()
        
    def forward(self, teacher_images, student_images, tau=None):
        emb_teachers = self.teacher_model(teacher_images)
        emb_students = self.student_model(student_images)
            
        if self.training:
            positive_pairs = (emb_teachers, emb_students)
            negative_pairs = self.curr_mining(positive_pairs, tau)
                
            return positive_pairs, negative_pairs
        
        else:
            return emb_teachers, emb_students

class CurriculumMining(nn.Module):

    def __init__(self):
        super(CurriculumMining, self).__init__()
        
    def forward(self, positive_pairs, tau):
        teacher_images, student_images = positive_pairs
        B, D = teacher_images.size()
        dists = torch.cdist(teacher_images, student_images, p=2)
        rev_dists = dists - torch.diag(torch.add(torch.diag(dists),1))
        sorted_dist, sorted_idx = torch.sort(rev_dists, dim=1, descending=True)
        
        idx_threshold = (B-1)-round(tau * (B-1))
        if idx_threshold == B-1:
            idx_threshold = B-2

        negative_sample_idx = sorted_idx[:, idx_threshold].view(B)        
        negative_student_images = student_images[negative_sample_idx]

        return teacher_images, negative_student_images

class TauScheduler(object):

    def __init__(self, lowest, highest): #lowest is 0.3, highest is 0.8
        self.current = int(lowest * 100)
        self.highest = int(highest * 100)
        self.epoch_num = 0

    def step(self):            
        if self.epoch_num % 2 == 0 and self.epoch_num > 0:
            self.current = int(self.current + self.current * 0.1)
        
        if self.current > self.highest:
            self.current = 80
    
        self.epoch_num += 1
        
    def get_tau(self):
        return self.current / 100

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size=BATCHSIZE):
        super(ContrastiveLoss, self).__init__()
        self.batchsize=batch_size
        self.pdist = torch.nn.PairwiseDistance(p=2)
    
    def forward(self, positive_pairs, negative_pairs, alpha):
        teacher_pos, student_pos = positive_pairs
        dists_pos = self.pdist(teacher_pos, student_pos)
        pos_part = dists_pos**2

        teacher_neg, student_neg = negative_pairs
        dists_neg = self.pdist(teacher_neg, student_neg)
        neg_part = (alpha-dists_neg).clamp(0) ** 2

        TBoard.add_scalar('Train/pos_part_sum', pos_part.sum().item(), step_num)
        TBoard.add_scalar('Train/neg_part_sum', neg_part.sum().item(), step_num)
        TBoard.add_scalar('Train/dists_neg_mean', dists_neg.mean().item(), step_num)
        TBoard.add_scalar('Train/dists_pos_mean', dists_pos.mean().item(), step_num)
        TBoard.add_scalar('Train/teacher_neg_mean', teacher_neg.mean().item(), step_num)
        TBoard.add_scalar('Train/student_neg_mean', student_neg.mean().item(), step_num)
        TBoard.add_scalar('Train/teacher_pos_mean', teacher_pos.mean().item(), step_num)
        TBoard.add_scalar('Train/student_neg_mean', student_neg.mean().item(), step_num)
        TBoard.add_scalar('Train/teacher_neg_student_neg_mean', (teacher_neg - student_neg).mean().item(), step_num)
        TBoard.add_scalar('Train/teacher_pos_student_pos_mean', (teacher_pos - student_pos).mean().item(), step_num)
        TBoard.add_scalar('Train/student_pos_student_neg_mean', (student_pos - student_neg).mean().item(), step_num)

        batch_loss = pos_part.sum() + neg_part.sum()
        batch_loss/=(self.batchsize+self.batchsize)
        return batch_loss

######################################################################

criterion = ContrastiveLoss(batch_size=BATCHSIZE)

model = CompleteDistillationModel()

train_img_dir = "/ssd_scratch/cvit/starc52/vggface2/"
train_csv_file = "/home/starc52/verandret/train_image_list.csv"
LOG_PATH = '/home/starc52/LearnablePINs/train_log_'+str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+"/"
CHECKPOINT_PATH = "/ssd_scratch/cvit/starc52/checkpoints/"
if os.path.isdir(CHECKPOINT_PATH):
    pass
else:
    os.makedirs(CHECKPOINT_PATH)


TBoard = tensorboardX.SummaryWriter(log_dir=LOG_PATH)

train = VoxCeleb2VGGface2Dataset(train_img_dir, train_csv_file, teacher_transform=teacher_transform, student_transform=student_transform)
trainloader = torch.utils.data.DataLoader(train, batch_size=BATCHSIZE, num_workers=NUM_WORKERS, shuffle=True)

LR_INIT = 1e-2
LR_LAST = 1e-8
gamma = 10 ** (np.log10(LR_LAST / LR_INIT) / (EPOCH_NUM - 1))

optimizer = torch.optim.SGD(model.parameters(), lr = LR_INIT, momentum=0.9, weight_decay=5e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
tau_scheduler = TauScheduler(lowest=0.3, highest=0.8)

device = torch.device('cuda')
model = model.to(device)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

for epoch_num in range(EPOCH_NUM):
    for iter_num, (teacher_images, student_images) in tqdm(enumerate(trainloader)):
        step_num = epoch_num * len(trainloader) + iter_num
        teacher_images, student_images = teacher_images.to(device, non_blocking=True), student_images.to(device, non_blocking=True)

        optimizer.zero_grad()

        positive_pairs, negative_pairs = model(teacher_images, student_images, tau=tau_scheduler.get_tau())

        loss = criterion(positive_pairs, negative_pairs, alpha=ALPHA)

        loss.backward()
        optimizer.step()

        TBoard.add_scalar('Train/Loss', loss.item(), step_num)
        TBoard.add_scalar('Train/lr', lr_scheduler.get_last_lr()[0], step_num)
        TBoard.add_scalar('Train/tau', tau_scheduler.get_tau(), step_num)
        TBoard.add_histogram('teacher_images', positive_pairs[0], step_num)
        TBoard.add_histogram('student_images', positive_pairs[1], step_num)

        if step_num%10000==0:
            torch.save({
                'epoch':epoch_num,
                'lr_scheduler':lr_scheduler.state_dict(),
                'tau_scheduler':tau_scheduler.get_tau(),
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss':loss,
            }, os.path.join(CHECKPOINT_PATH, str(step_num)+".pth"))
    lr_scheduler.step()
    tau_scheduler.step()


