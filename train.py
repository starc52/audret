import sys
sys.path.append('../')
from utils.utils import *
import time
import numpy as np
import os
from os.path import join
from tqdm import tqdm

# class Trainer(nn.Module):

# 	def __init__(
# 				self,
# 				model,
# 				optimizer,
# 				loader,
# 				criterion,
# 				args,
# 				fn_name='contrastive'
# 			):
# 		super(Trainer, self).__init__()

# 		self.model = model
# 		self.optimizer = optimizer
# 		self.loader = loader
# 		self.criterion = criterion
# 		self.fn_name = fn_name
# 		self.args = args

# 	def forward(epoch):


def train_softmax(model, optimizer, loader, epoch, device, args, criterion):
	model.train()

	tic = time.time()
	
	# criterion_cosine = nn.CosineSimilarity()
	total_loss = AverageMeter()
	# sim_loss_centroid = AverageMeter()


	# global_speaker_centroid = torch.from_numpy(np.load(join(args.root, 'vgg_vox_embeddings', 'vgg_vox_train_centroid.npy'))).view(1,-1).to(device)


	for idx, (face_embedding, speaker_embeddings) in enumerate(loader):

		face_embedding = face_embedding.to(device)
		speaker_embeddings = speaker_embeddings.to(device)

		optimizer.zero_grad()

		assert speaker_embeddings.shape[0] == face_embedding.shape[0]

		pred_face_embedding = model(face_embedding)
		# print(speaker_embeddings.shape, pred_face_embedding.shape, label.shape)

		scores = torch.bmm(speaker_embeddings, pred_face_embedding.unsqueeze(2)).squeeze(2)

		loss = criterion(scores, label)

		# loss = args.kldiv_coeff * criterion_kldiv(pred_embedding, speaker_embedding) + \
							# args.ge2e_coeff * criterion_ge2e(pred_embedding.view(-1, 10, 2048)) + \
								# args.mse_coeff * criterion_mse(pred_embedding, speaker_embedding)

		loss.backward()
		optimizer.step()

		total_loss.update(float(loss))
		# sim_loss_centroid.update(float(criterion_cosine(pred_embedding, global_speaker_centroid.repeat(pred_embedding.shape[0], 1)).mean()))

	print('[{:2d}, train] avg_loss : {:.5f} time:{:3f} minutes'.format(epoch, total_loss.avg, (time.time()-tic)/60))
	sys.stdout.flush()

	return total_loss.avg

def validate_softmax(model, loader, epoch, device, args, criterion):
	model.eval()

	tic = time.time()
	
	# criterion_cosine = nn.CosineSimilarity()
	total_loss = AverageMeter()
	# sim_loss_centroid = AverageMeter()


	# global_speaker_centroid = torch.from_numpy(np.load(join(args.root, 'vgg_vox_embeddings', 'vgg_vox_train_centroid.npy'))).view(1,-1).to(device)

	correct = 0
	for idx, (face_embedding, speaker_embeddings, label) in enumerate(loader):

		face_embedding = face_embedding.to(device)
		speaker_embeddings = speaker_embeddings.to(device)
		label = label.to(device)

		assert speaker_embeddings.shape[0] == face_embedding.shape[0]

		pred_face_embedding = model(face_embedding)

		scores = torch.bmm(speaker_embeddings, pred_face_embedding.unsqueeze(2)).squeeze(2)

		loss = criterion(scores, label)

		_, pred = torch.max(scores, 1)
		correct += (pred == label).sum().item()


		# loss = args.kldiv_coeff * criterion_kldiv(pred_embedding, speaker_embedding) + \
							# args.ge2e_coeff * criterion_ge2e(pred_embedding.view(-1, 10, 2048)) + \
								# args.mse_coeff * criterion_mse(pred_embedding, speaker_embedding)

		total_loss.update(float(loss))
		# sim_loss_centroid.update(float(criterion_cosine(pred_embedding, global_speaker_centroid.repeat(pred_embedding.shape[0], 1)).mean()))

	print('[{:2d}, train] avg_loss : {:.5f}, accuracy : {:.5f}, time:{:3f} minutes'.format(epoch, total_loss.avg, correct / len(loader), (time.time()-tic)/60))
	sys.stdout.flush()

	return total_loss.avg


''' Fix GE2E criterion and face embedding shape in model '''
def train(model, optimizer, loader, epoch, device, args, criterion, tau):
	model.train()

	tic = time.time()
	
	criterion_cosine = nn.CosineSimilarity()
	total_loss = AverageMeter()
	sim_loss_centroid = AverageMeter()


	# global_speaker_centroid = torch.from_numpy(np.load(join(args.root, 'vgg_vox_embeddings', 'vgg_vox_train_centroid.npy'))).view(1,-1).to(device)


	for idx, (face_embedding, speaker_embedding, _id) in enumerate(loader):

		face_embedding = face_embedding.to(device)
		speaker_embedding = speaker_embedding.to(device)

		optimizer.zero_grad()

		# face_embedding = face_embedding.view(-1, 2048)
		# speaker_embedding = speaker_embedding.view(-1, 4096)

		assert speaker_embedding.shape[0] == face_embedding.shape[0]

		pred_face_embedding, pred_speaker_embedding = model(face_embedding, speaker_embedding)

		loss = criterion(pred_face_embedding, pred_speaker_embedding, tau=tau)

		# loss = args.kldiv_coeff * criterion_kldiv(pred_embedding, speaker_embedding) + \
							# args.ge2e_coeff * criterion_ge2e(pred_embedding.view(-1, 10, 2048)) + \
								# args.mse_coeff * criterion_mse(pred_embedding, speaker_embedding)

		loss.backward()
		optimizer.step()

		total_loss.update(float(loss))
		# sim_loss_centroid.update(float(criterion_cosine(pred_embedding, global_speaker_centroid.repeat(pred_embedding.shape[0], 1)).mean()))

	print('[{:2d}, train] avg_loss : {:.5f}, centroid sim score : {:.5f}, time:{:3f} minutes'.format(epoch, total_loss.avg, sim_loss_centroid.avg, (time.time()-tic)/60))
	sys.stdout.flush()

	return total_loss.avg


def validate(model, loader, epoch, device, args, criterion, tau):
	model.eval()

	tic = time.time()
	
	total_loss = AverageMeter()

	criterion_cosine = nn.CosineSimilarity()
	sim_loss = AverageMeter()
	sim_loss_centroid = AverageMeter()

	# global_speaker_centroid = torch.from_numpy(np.load(join(args.root, 'vgg_vox_embeddings', 'vgg_vox_test_centroid.npy'))).view(1,-1).to(device)

	for idx, (face_embedding, speaker_embedding, _id) in enumerate(loader):

		face_embedding = face_embedding.to(device)
		speaker_embedding = speaker_embedding.to(device)

		# face_embedding = face_embedding.view(-1, 2048)
		# speaker_embedding = speaker_embedding.view(-1, 4096)

		assert speaker_embedding.shape[0] == face_embedding.shape[0]

		pred_face_embedding, pred_speaker_embedding = model(face_embedding, speaker_embedding)

		loss = criterion(pred_face_embedding, pred_speaker_embedding, tau=tau)
		
		# loss = args.kldiv_coeff * criterion_kldiv(pred_embedding, speaker_embedding) + \
							# args.ge2e_coeff * criterion_ge2e(pred_embedding.view(-1, 10, 2048)) + \
								# args.mse_coeff * criterion_mse(pred_embedding, speaker_embedding)

		total_loss.update(float(loss))

		sim_loss.update(float(criterion_cosine(pred_face_embedding, pred_speaker_embedding).mean()))
		# sim_loss_centroid.update(float(criterion_cosine(pred_embedding, global_speaker_centroid.repeat(pred_embedding.shape[0], 1)).mean()))

	print('[{:2d}, eval] avg_loss : {:.5f} cos-sim : {:.5f}, centroid sim score : {:.5f}, time:{:3f} minutes'.format(epoch, total_loss.avg, sim_loss.avg, sim_loss_centroid.avg, (time.time()-tic)/60))
	sys.stdout.flush()

	return total_loss.avg














def train_single(model, optimizer, loader, epoch, device, args, criterion, tau):
	model.train()

	tic = time.time()
	
	criterion_cosine = nn.CosineSimilarity()
	total_loss = AverageMeter()
	sim_loss_centroid = AverageMeter()


	# global_speaker_centroid = torch.from_numpy(np.load(join(args.root, 'vgg_vox_embeddings', 'vgg_vox_train_centroid.npy'))).view(1,-1).to(device)


	for idx, (face_embedding, speaker_embedding, _id) in enumerate(loader):

		face_embedding = face_embedding.to(device)
		speaker_embedding = speaker_embedding.to(device)

		optimizer.zero_grad()

		# face_embedding = face_embedding.view(-1, 2048)
		# speaker_embedding = speaker_embedding.view(-1, 4096)

		assert speaker_embedding.shape[0] == face_embedding.shape[0]

		pred_face_embedding = model(face_embedding)

		loss = criterion(pred_face_embedding, speaker_embedding, tau=tau)

		# loss = args.kldiv_coeff * criterion_kldiv(pred_embedding, speaker_embedding) + \
							# args.ge2e_coeff * criterion_ge2e(pred_embedding.view(-1, 10, 2048)) + \
								# args.mse_coeff * criterion_mse(pred_embedding, speaker_embedding)

		loss.backward()
		optimizer.step()

		total_loss.update(float(loss))
		# sim_loss_centroid.update(float(criterion_cosine(pred_embedding, global_speaker_centroid.repeat(pred_embedding.shape[0], 1)).mean()))

	print('[{:2d}, train] avg_loss : {:.5f}, centroid sim score : {:.5f}, time:{:3f} minutes'.format(epoch, total_loss.avg, sim_loss_centroid.avg, (time.time()-tic)/60))
	sys.stdout.flush()

	return total_loss.avg


def validate_single(model, loader, epoch, device, args, criterion, tau):
	model.eval()

	tic = time.time()
	
	total_loss = AverageMeter()

	criterion_cosine = nn.CosineSimilarity()
	sim_loss = AverageMeter()
	sim_loss_centroid = AverageMeter()

	# global_speaker_centroid = torch.from_numpy(np.load(join(args.root, 'vgg_vox_embeddings', 'vgg_vox_test_centroid.npy'))).view(1,-1).to(device)

	for idx, (face_embedding, speaker_embedding, _id) in enumerate(loader):

		face_embedding = face_embedding.to(device)
		speaker_embedding = speaker_embedding.to(device)

		# face_embedding = face_embedding.view(-1, 2048)
		# speaker_embedding = speaker_embedding.view(-1, 4096)

		assert speaker_embedding.shape[0] == face_embedding.shape[0]

		pred_face_embedding = model(face_embedding)

		loss = criterion(pred_face_embedding, speaker_embedding, tau=tau)
		
		# loss = args.kldiv_coeff * criterion_kldiv(pred_embedding, speaker_embedding) + \
							# args.ge2e_coeff * criterion_ge2e(pred_embedding.view(-1, 10, 2048)) + \
								# args.mse_coeff * criterion_mse(pred_embedding, speaker_embedding)

		total_loss.update(float(loss))

		sim_loss.update(float(criterion_cosine(pred_face_embedding, speaker_embedding).mean()))
		# sim_loss_centroid.update(float(criterion_cosine(pred_embedding, global_speaker_centroid.repeat(pred_embedding.shape[0], 1)).mean()))

	print('[{:2d}, eval] avg_loss : {:.5f} cos-sim : {:.5f}, centroid sim score : {:.5f}, time:{:3f} minutes'.format(epoch, total_loss.avg, sim_loss.avg, sim_loss_centroid.avg, (time.time()-tic)/60))
	sys.stdout.flush()

	return total_loss.avg
