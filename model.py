import torch
import torch.nn as nn
from loss import *
from train import train

# class Face2Speaker(nn.Module):

#     def __init__(self, embedding_dim=4096):

#         super(Face2Speaker, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(2048, embedding_dim),
#             nn.ReLU(inplace=True),
#             L2Norm()
#         )


#     def forward(self, x):
#         batch_size = x.size(0)
#         x = x.view(batch_size, -1)
#         x = self.fc(x)
#         return x

# class SingleLayerModel(nn.Module):

#     def __init__(self, embedding_dim=4096):

#         super(SingleLayerModel, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(2048, embedding_dim),
#             nn.ReLU(inplace=True),
#             L2Norm()
#         )

#     def forward(self, face):
#         batch_size = face.size(0)
        
#         face = self.fc(face)
#         return face

class LearnablePINSenetVggVox256(nn.Module):

    def __init__(self, embedding_dim=256, training=True):

        super(LearnablePINSenetVggVox256, self).__init__()
        self.face_fc = nn.Sequential(
            nn.Linear(2048, embedding_dim),
            nn.ReLU(inplace=True),
            L2Norm()
        )

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
        
        face = self.face_fc(face)
        audio = self.audio_fc(audio)
        if self.training:
            positive_pairs = (face, audio)
            negative_pairs = self.curr_mining(positive_pairs, tau)
                
            return positive_pairs, negative_pairs
        else:
            return face, audio


class L2Norm(nn.Module):
	def __init__(self):
		super(L2Norm, self).__init__()

	def forward(self, x):
		x = x / torch.norm(x, dim=1, keepdim=True)
		return x