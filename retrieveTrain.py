from posixpath import split
import sys
sys.path.append('../')
import torch
import torch.nn as nn
import numpy as np 
import os
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from os.path import join
import argparse
import random
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import *
from dataloader import *
from utils.utils import *
from loss import *
from tqdm import tqdm
#from train import *
from datetime import datetime
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs',default=50, type=int)
parser.add_argument('--batch_size',default=160, type=int)
parser.add_argument('--alpha',default=0.6, type=float)
parser.add_argument('--log_interval',default=20, type=int)
parser.add_argument('--no_workers',default=36, type=int)
parser.add_argument('--compress_train', action='store_true')
parser.add_argument('--loss_factor', default=0.25, type=float)
parser.add_argument('--log_path',default='/home/starc52/LearnablePINs/train_log_'+str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))+"/", type=str)
parser.add_argument('--checkpoint_path',default="/ssd_scratch/cvit/starc52/LPscheckpoints/", type=str)
parser.add_argument('--root',default='/ssd_scratch/cvit/starc52/VoxCeleb2/dev/mp4/', type=str)

args = parser.parse_args()
print(args)
os.makedirs(args.log_path, exist_ok=True)
os.makedirs(args.checkpoint_path, exist_ok=True)

######################################################################

criterion = ContrastiveLoss(batch_size=args.batch_size, alpha=args.alpha)

model = LearnablePINSenetVggVox256()

LOG_PATH = args.log_path
CHECKPOINT_PATH = args.checkpoint_path

TBoard = SummaryWriter(log_dir=args.log_path)
if args.compress_train:
    traindataset = simpleDataLoaderLossy(args.loss_factor, args.root, split='train')
else:
    traindataset = simpleDataLoader(args.root, split='train')

trainloader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, num_workers=args.no_workers, shuffle=True)

LR_INIT = 1e-2
LR_LAST = 1e-8
gamma = 10 ** (np.log10(LR_LAST / LR_INIT) / (args.num_epochs - 1))

optimizer = torch.optim.SGD(model.parameters(), lr = LR_INIT, momentum=0.9, weight_decay=5e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
tau_scheduler = TauScheduler(lowest=0.3, highest=0.8)

device = torch.device('cuda')
model = model.to(device)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

for epoch_num in tqdm(range(args.num_epochs)):
    for iter_num, (face, audio) in enumerate(trainloader):
        step_num = epoch_num * len(trainloader) + iter_num
        face, audio = face.to(device, non_blocking=True), audio.to(device, non_blocking=True)

        optimizer.zero_grad()

        positive_pairs, negative_pairs = model(face, audio, tau=tau_scheduler.get_tau())

        loss = criterion(positive_pairs, negative_pairs)

        loss.backward()
        optimizer.step()

        TBoard.add_scalar('Train/Loss', loss.item(), step_num)
        TBoard.add_scalar('Train/lr', lr_scheduler.get_last_lr()[0], step_num)
        TBoard.add_scalar('Train/tau', tau_scheduler.get_tau(), step_num)
        # TBoard.add_histogram('teacher_images', positive_pairs[0], step_num)
        # TBoard.add_histogram('student_images', positive_pairs[1], step_num)

    torch.save({
        'epoch':epoch_num,
        'lr_scheduler':lr_scheduler.state_dict(),
        'tau_scheduler':tau_scheduler.get_tau(),
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'loss':loss,
    }, os.path.join(CHECKPOINT_PATH, "model_e"+str(epoch_num)+".pth"))
    lr_scheduler.step()
    tau_scheduler.step()

