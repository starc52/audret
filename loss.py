import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# class ContrastiveLoss(torch.nn.Module):
#     """
#     Contrastive loss function.
#     Based on:
#     """

#     def __init__(self, margin=1.0):
#         super(ContrastiveLoss, self).__init__()
#         print("Margin :", margin)
#         self.margin = margin

#     def check_type_forward(self, in_types):
#         assert len(in_types) == 3

#         x0_type, x1_type, y_type = in_types
#         assert x0_type.size() == x1_type.shape
#         assert x1_type.size()[0] == y_type.shape[0]
#         assert x1_type.size()[0] > 0
#         assert x0_type.dim() == 2
#         assert x1_type.dim() == 2
#         assert y_type.dim() == 1

#     def forward(self, x0, x1, y):
#         self.check_type_forward((x0, x1, y))

#         # euclidian distance
#         diff = x0 - x1
#         dist_sq = torch.sum(torch.pow(diff, 2), 1)
#         dist = torch.sqrt(dist_sq)

#         mdist = self.margin - dist
#         dist = torch.clamp(mdist, min=0.0)
#         loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
#         loss = torch.sum(loss) / 2.0 / x0.size()[0]
#         return loss

# class ContrastiveWithHardNegative(nn.Module):

#     def __init__(self, margin=1.0):
#         super(ContrastiveWithHardNegative, self).__init__()
#         self.margin = margin
#         self.constrastive_loss = ContrastiveLoss(margin=self.margin)
#         self.negative_miner = CurriculumNegativeMining()

#     def forward(self, prediction, speaker, tau=None):
#         # if prediction.requires_grad == False:
#         #     return self.constrastive_loss(prediction, speaker, torch.ones(prediction.shape[0], ).cuda())
#         neg = self.negative_miner(prediction, speaker, tau=tau)
#         return self.constrastive_loss(prediction, speaker, torch.ones(prediction.shape[0], ).cuda()) +\
#                 self.constrastive_loss(prediction, neg, torch.zeros(prediction.shape[0], ).cuda()) 
        

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, alpha):
        super(ContrastiveLoss, self).__init__()
        self.batchsize=batch_size
        self.alpha=alpha
        self.pdist = torch.nn.PairwiseDistance(p=2)
    
    def forward(self, positive_pairs, negative_pairs):
        face_pos, audio_pos = positive_pairs
        dists_pos = self.pdist(face_pos, audio_pos)
        pos_part = dists_pos**2

        face_neg, audio_neg = negative_pairs
        dists_neg = self.pdist(face_neg, audio_neg)
        neg_part = (self.alpha-dists_neg).clamp(0) ** 2

        batch_loss = pos_part.sum() + neg_part.sum()
        batch_loss/=(2*self.batchsize)
        return batch_loss


class CurriculumMining(nn.Module):

    def __init__(self):
        super(CurriculumMining, self).__init__()
        
    def forward(self, positive_pairs, tau):
        teacher_images, student_images = positive_pairs
        B, D = teacher_images.size()
        dists = torch.cdist(teacher_images, student_images, p=2)
        rev_dists = dists - torch.diag(torch.add(torch.diag(dists),1))
        sorted_dist, sorted_idx = torch.sort(rev_dists, dim=1, descending=True)
        
        idx_threshold = round(tau * (B-1))-1
        if idx_threshold < 0:
            idx_threshold = 0

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


# class CurriculumNegativeMining(nn.Module):

#     def __init__(self):
#         super(CurriculumNegativeMining, self).__init__()

#     def forward(self, face, audio, tau=None):

#         batch_size = face.size(0)

#         assert (batch_size-1) * tau > 0.5

#         dists = torch.cdist(face, audio)
#         dists_copy = dists + torch.diag(np.inf * torch.ones((batch_size, ))).cuda()

#         dists_ranked, sorted_indices = torch.sort(dists_copy, dim=1, descending=True)

#         n_tau = int(np.round(tau * (batch_size-1)))
#         n_i = torch.argmin(torch.abs(dists_ranked - torch.diag(dists).unsqueeze(1)), dim=1, keepdim=True)

#         p_i = torch.min(n_i, n_tau * torch.ones_like(n_i)).view(-1,)

#         ind = sorted_indices.gather(1, p_i.view(-1, 1)).view(-1,)
#         # temp_ind = dists_ranked.gather(1, p_i.view(-1, 1))
#         x_neg = audio[ind]

#         # print(sorted_indices)
#         # print(p_i)
#         # print(ind) 
#         # print(temp_ind)

#         return x_neg 

# class GE2ELoss(nn.Module):

#     def __init__(self, init_w=10.0, init_b=-5.0, loss_method='softmax'):
#         '''
#         Implementation of the Generalized End-to-End loss defined in https://arxiv.org/abs/1710.10467 [1]
#         Accepts an input of size (N, M, D)
#             where N is the number of speakers in the batch,
#             M is the number of utterances per speaker,
#             and D is the dimensionality of the embedding vector (e.g. d-vector)
#         Args:
#             - init_w (float): defines the initial value of w in Equation (5) of [1]
#             - init_b (float): definies the initial value of b in Equation (5) of [1]
#         '''
#         super(GE2ELoss, self).__init__()
#         self.w = nn.Parameter(torch.tensor(init_w))
#         self.b = nn.Parameter(torch.tensor(init_b))
#         self.loss_method = loss_method

#         assert self.loss_method in ['softmax', 'contrast']

#         if self.loss_method == 'softmax':
#             self.embed_loss = self.embed_loss_softmax
#         if self.loss_method == 'contrast':
#             self.embed_loss = self.embed_loss_contrast

#     def calc_new_centroids(self, dvecs, centroids, spkr, utt):
#         '''
#         Calculates the new centroids excluding the reference utterance
#         '''
#         excl = torch.cat((dvecs[spkr,:utt], dvecs[spkr,utt+1:]))
#         excl = torch.mean(excl, 0)
#         new_centroids = []
#         for i, centroid in enumerate(centroids):
#             if i == spkr:
#                 new_centroids.append(excl)
#             else:
#                 new_centroids.append(centroid)
#         return torch.stack(new_centroids)

#     def calc_cosine_sim(self, dvecs, centroids):
#         '''
#         Make the cosine similarity matrix with dims (N,M,N)
#         '''
#         cos_sim_matrix = []
#         for spkr_idx, speaker in enumerate(dvecs):
#             cs_row = []
#             for utt_idx, utterance in enumerate(speaker):
#                 new_centroids = self.calc_new_centroids(dvecs, centroids, spkr_idx, utt_idx)
#                 # vector based cosine similarity for speed
#                 cs_row.append(torch.clamp(torch.mm(utterance.unsqueeze(1).transpose(0,1), new_centroids.transpose(0,1)) / (torch.norm(utterance) * torch.norm(new_centroids, dim=1)), 1e-6))
#             cs_row = torch.cat(cs_row, dim=0)
#             cos_sim_matrix.append(cs_row)
#         return torch.stack(cos_sim_matrix)

#     def embed_loss_softmax(self, dvecs, cos_sim_matrix):
#         '''
#         Calculates the loss on each embedding $L(e_{ji})$ by taking softmax
#         '''
#         N, M, _ = dvecs.shape
#         L = []
#         for j in range(N):
#             L_row = []
#             for i in range(M):
#                 L_row.append(-F.log_softmax(cos_sim_matrix[j,i], 0)[j])
#             L_row = torch.stack(L_row)
#             L.append(L_row)
#         return torch.stack(L)

#     def embed_loss_contrast(self, dvecs, cos_sim_matrix):
#         ''' 
#         Calculates the loss on each embedding $L(e_{ji})$ by contrast loss with closest centroid
#         '''
#         N, M, _ = dvecs.shape
#         L = []
#         for j in range(N):
#             L_row = []
#             for i in range(M):
#                 centroids_sigmoids = torch.sigmoid(cos_sim_matrix[j,i])
#                 excl_centroids_sigmoids = torch.cat((centroids_sigmoids[:j], centroids_sigmoids[j+1:]))
#                 L_row.append(1. - torch.sigmoid(cos_sim_matrix[j,i,j]) + torch.max(excl_centroids_sigmoids))
#             L_row = torch.stack(L_row)
#             L.append(L_row)
#         return torch.stack(L)

#     def forward(self, dvecs):
#         '''
#         Calculates the GE2E loss for an input of dimensions (num_speakers, num_utts_per_speaker, dvec_feats)
#         '''
#         #Calculate centroids
#         centroids = torch.mean(dvecs, 1)

#         #Calculate the cosine similarity matrix
#         cos_sim_matrix = self.calc_cosine_sim(dvecs, centroids)
#         torch.clamp(self.w, 1e-6)
#         cos_sim_matrix = cos_sim_matrix * self.w + self.b
#         L = self.embed_loss(dvecs, cos_sim_matrix)
#         return L.sum()



# class TauScheduler(object):
#     '''
#     "found that it was effective to increase \tau by 10 percent 
#     every two epochs, starting from 30% up until 80%, and keeping 
#     it constant thereafter"
#     --- So, it is increasing by 10 % every second epoch:
#             ⎧tau = tau + tau * 0.1, tau < 0.8, 
#             ⎨
#             ⎩tau = 0.8, tau > 0.8.
#     '''
    
#     def __init__(self, lowest, highest):
#         self.current = int(lowest * 100)
#         self.highest = int(highest * 100)
#         self.epoch_num = 0

#     def step(self):
            
#         if self.epoch_num % 2 == 0 and self.epoch_num > 0:
# #         if self.epoch_num % 20 == 0 and self.epoch_num > 0: # here  
# #                 self.current += 10
#             self.current = int(self.current + self.current * 0.1)
        
#         if self.current > self.highest:
#             self.current = 80
    
#         self.epoch_num += 1
        
#     def get_tau(self):
# #         return np.random.uniform() # here
#         return self.current / 100
