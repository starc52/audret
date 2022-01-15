import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        
        hard_semi_negative = torch.argmin(torch.abs((dists+torch.diag(torch.add(-1*torch.diag(dists),float('inf')))) - torch.unsqueeze(torch.diag(dists), 1)), dim=1)

        negative_sample_idx = sorted_idx[:, idx_threshold].view(B)
        fin_idx = torch.min(hard_semi_negative, negative_sample_idx)
        negative_student_images = student_images[fin_idx]

        return teacher_images, negative_student_images

class TauScheduler(object):

    def __init__(self, lowest, highest): #lowest is 0.3, highest is 0.8
        self.current = int(lowest * 100)
        self.highest = int(highest * 100)
        self.epoch_num = 0

    def step(self):            
        if self.epoch_num % 3 == 0 and self.epoch_num > 0:
            self.current = int(self.current + self.current * 0.1)
        
        if self.current > self.highest:
            self.current = 80
    
        self.epoch_num += 1
        
    def get_tau(self):
        return self.current / 100
