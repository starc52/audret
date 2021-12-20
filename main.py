import sys
sys.path.append('../')
import torch
import torch.nn as nn
import numpy as np 
import os
from torch.utils.data import Dataset, DataLoader
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
from train import *

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs',default=30, type=int)
parser.add_argument('--lr',default=1e-4, type=float)
parser.add_argument('--batch_size',default=32, type=int)
parser.add_argument('--log_interval',default=20, type=int)
parser.add_argument('--no_workers',default=10, type=int)
parser.add_argument('--model_val_path',default="face2speaker_sevggvox_mse_ge2e.pt", type=str)
parser.add_argument('--kldiv_coeff',default=0, type=float)
parser.add_argument('--ge2e_coeff',default=0, type=float)
parser.add_argument('--mse_coeff',default=0, type=float)
parser.add_argument('--l1_coeff',default=0, type=float)
parser.add_argument('--root',default='/ssd_scratch/cvit/samyak/', type=str)
parser.add_argument('--plot_name',default='plot_e1.png', type=str)
parser.add_argument('--margin',default=0.6, type=float)

args = parser.parse_args()
print(args)

os.makedirs('saved_models', exist_ok=True)

model = LearnablePINSenetVggVox256(embedding_dim=256)
# model = SingleLayerModel()

for (name, param) in model.named_parameters():
	if param.requires_grad:
		print(name, param.size())

# train_dataset = ShuffledPositiveUtteranceEmbeddingLoader(
# 				face_embed_root = join(args.root,'senet_face_embed_full_data', 'train'),
# 				speaker_embed_root = join(args.root,'voxceleb2_vgg_vox_embeddings','train'),
# 				split='train'
# 			)

# val_dataset = ShuffledPositiveUtteranceEmbeddingLoader(
# 				face_embed_root = join(args.root,'senet_face_embed_full_data', 'test'),
# 				speaker_embed_root = join(args.root,'voxceleb2_vgg_vox_embeddings','test'),
# 				split='val'
# 			)

train_dataset = simpleDataLoader(root = args.root, split='train')

val_dataset = simpleDataLoader(root = args.root,split='val')


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.no_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device, len(train_loader), len(val_loader))

if torch.cuda.device_count() > 1:
	print("Let's use", torch.cuda.device_count(), "GPUs!")
	model = nn.DataParallel(model)

model.to(device)

# criterion_ge2e = GE2ELoss().to(device)
# criterion_kldiv = nn.KLDivLoss()
# criterion_mse = nn.MSELoss()

criterion = ContrastiveWithHardNegative(margin=args.margin).to(device)

criterion_softmax = nn.CrossEntropyLoss(reduction='mean')
# def get_tau(start, end, num_epochs):
# 	tau_arr = np.round(np.linspace(start, end, num_epochs), 1)	
# 	return tau_arr

# tau_arr = get_tau(0.3, 0.8, args.num_epochs)
# print(tau_arr)

tau_sched = TauScheduler(0.3, 0.8)

# model_params = list(filter(lambda p: p.requires_grad, model.parameters()))
params = list(filter(lambda p: p.requires_grad, model.parameters()))
# ge2e_params = criterion_ge2e.parameters()

# params = [{"params": model_params}, {"params": ge2e_params}]

optimizer = torch.optim.Adam(params, lr=args.lr)

best_loss = np.inf

train_loss_arr = []
val_loss_arr = []

for epoch in range(args.num_epochs):
	tau_sched.step()
	print(tau_sched.get_tau())
	# loss = train(model, optimizer, train_loader, epoch, device, args, criterion, tau=tau_sched.get_tau())
	loss = train_softmax(model, optimizer, train_loader, epoch, device, args, criterion_softmax)
	train_loss_arr.append(loss)
	with torch.no_grad():
		# val_loss = validate(model, val_loader, epoch, device, args, criterion, tau=tau_sched.get_tau())
		val_loss = validate_softmax(model, val_loader, epoch, device, args, criterion_softmax)
		val_loss_arr.append(val_loss)

		# if val_loss <= best_loss:
		# 	best_loss = val_loss
		# 	print('[{:2d},  save, {}]'.format(epoch, join('saved_models', args.model_val_path)))
		# 	print("="*50)
		if torch.cuda.device_count() > 1:  
			torch.save(model.module.state_dict(), join('saved_models', args.model_val_path))
		else:
			torch.save(model.state_dict(), join('saved_models', args.model_val_path))

	print()

	if epoch%1==0:
		fig = plt.figure()
		plt.plot(np.arange(0,len(train_loss_arr)),train_loss_arr,'b-',label='train loss')
		plt.plot(np.arange(0,len(val_loss_arr)),val_loss_arr,'r-',label='val loss')
		plt.title('Epoch :%d'%(epoch))
		plt.grid()
		plt.legend()
		plt.savefig('./plots/loss_{}'.format(args.plot_name))
		print("Loss plot saved")
		print("="*20)
