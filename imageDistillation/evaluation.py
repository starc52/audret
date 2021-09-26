import numpy as np
import os,sys
sys.path.append('../')
from os.path import join
import glob
import  random
import scipy
from scipy import spatial
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
# from face2speaker_main import Face2Speaker
# from Recode.model import DoubleLayerModel, SingleLayerModel
from senet import *
from PIL import Image
from torchvision import transforms
from time import time

random.seed(0)
# random.seed(time())

class Evaluation():
    def __init__(self, 
                face_embedding_path, 
                speaker_embedding_path, 
                embedder,
                num_queries=100, 
                gallery_size=5, 
                mode='ff',
                distance_metric='euclidean',
                using_face_embeddings=False):

        self.embedder = embedder
        self.face_embedding_path = face_embedding_path
        self.speaker_embedding_path = speaker_embedding_path
        self.gallery_size = gallery_size
        self.num_queries = num_queries
        self.mode = mode
        self.distance_metric= distance_metric
        self.using_face_embeddings = using_face_embeddings
        self.generate_eval_data()
        pass

    def distance(self,a,b):
        if self.distance_metric=='euclidean':
            return np.linalg.norm(a-b)
        elif self.distance_metric=='cosine':
            return spatial.distance.cosine(a,b)

    def generate_eval_data(self):
        if self.mode=='fv':
            query_path = self.face_embedding_path
            gallery_path = self.speaker_embedding_path
        elif self.mode=='vf':
            query_path = self.speaker_embedding_path
            gallery_path = self.face_embedding_path
        if self.mode=='ff':
            query_path = self.face_embedding_path
            gallery_path = self.face_embedding_path
        elif self.mode=='vv':
            query_path = self.speaker_embedding_path
            gallery_path = self.speaker_embedding_path

        _id_list = sorted(os.listdir(query_path))

        queries = []
        final_gallery = []

        for _id in sorted(os.listdir(query_path)):
            _id_path = join(query_path, _id)
            for _url in sorted(os.listdir(_id_path)):
                _url_path = join(_id_path,_url)

                if self.using_face_embeddings:
                    for embedding in sorted(os.listdir(_url_path)):
                        emb_path = join(_url_path,embedding)
                        queries.append(emb_path)
                else:
                    if os.path.exists(join(_url_path,'frames')):
                        for embedding in sorted(os.listdir(join(_url_path,'frames'))):
                            emb_path = join(_url_path,'frames', embedding)
                            queries.append(emb_path)

        
        for _idx, query in enumerate(queries[0:10]):
            if self.using_face_embeddings:
                _id = query.split(os.sep)[-3]
            else:
                _id = query.split(os.sep)[-4]

            same_flag = 1

            while same_flag:
                answer_set = glob.glob(join(gallery_path, _id, join('*','*.npy')))

                answer = random.choice(answer_set)
                if(not answer.split(os.sep)[-1]==query.split(os.sep)[-1]):
                    same_flag = 0
                
            diff_speakers = [i for i in _id_list if i!=_id]
            random.shuffle(diff_speakers)

            assert not _id in diff_speakers

            impostor_gallery = [] 
            
            for imp in diff_speakers[0:self.gallery_size-1]:
                imp_embeddings = glob.glob(join(gallery_path,imp,join('*','*.npy')))
                impostor_gallery.append(random.choice(imp_embeddings))
                
            impostor_gallery.append(answer)
            final_gallery.append(impostor_gallery)
            
        self.queries = np.array(queries[0:self.num_queries])
        self.galleries = np.array(final_gallery[0:self.num_queries])
        self.answer = np.array([self.gallery_size-1]*self.num_queries)

        print("Num queries : %d"%(len(self.queries)))
        print("Gallery Size : %d"%(self.galleries.shape[1]))
        pass

    def evaluate(self):

        test_samples = self.num_queries

        result = []
        for _idx, query in enumerate(tqdm(self.queries[0:test_samples])):
            
            if self.using_face_embeddings:
                q_emb = self.embedder.get_embedding(input_path=query, mode=self.mode[0])
            else:
                q_emb = self.embedder.get_face_embedding(input_path=query, mode=self.mode[0])

            for gallery in self.galleries[0:test_samples]:
                distances = []
                for g in gallery:
                    g_emb = self.embedder.get_embedding(input_path=g, mode=self.mode[1])
                    distances.append(self.distance(q_emb,g_emb))

            result.append(np.argmin(distances))
        
        result = np.array(result)
        r = len(np.where(result==self.answer[0:test_samples])[0])
        accuracy = r/test_samples
        print("Identification Accuracy : %.4f"%(accuracy))
        return accuracy

class GetEmbeddings():
    def __init__(self, 
                 face_model,
                 speech_model):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.face_model = face_model 
        self.face_model.to(self.device)
        self.face_model.eval()

        self.speech_model = speech_model 
        if(speech_model!='na'):
            self.speech_model.to(self.device)
            self.speech_model.eval()
        
    def get_embedding(self, input_path=None, emb=None, mode='f'):
        if input_path is not None:
            emb = np.load(input_path).reshape(-1)
        emb /= np.linalg.norm(emb)
        emb = torch.from_numpy(emb).unsqueeze(0)
        emb = emb.to(self.device)
        with torch.no_grad():
            if mode == 'f':
                res_emb = self.face_model(emb)
            elif mode == 'v':
                if self.speech_model != 'na':
                    res_emb = self.speech_model(emb)
                else:
                    res_emb = emb

        res_emb = res_emb.cpu().numpy().reshape(-1)
        return res_emb

    def get_face_embedding(self, input_path=None, emb=None, mode='f'):
        input_path = input_path.replace('npy','png')
        self.img_transform = transforms.Compose([
                                                transforms.Resize((224, 224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(
                                                        [0.5, 0.5, 0.5],
                                                        [0.5, 0.5, 0.5]
                                                        )
                                                ])

        face_input = Image.open(input_path).convert('RGB')
        face_input = self.img_transform(face_input)
        face_input = face_input.to(self.device)
        face_input = face_input.unsqueeze(0)

        # emb = torch.from_numpy(face_input).unsqueeze(0)
        # emb = emb.to(self.device)
        
        with torch.no_grad():
            res_emb = self.face_model(face_input)

        res_emb = res_emb.cpu().numpy().reshape(-1)
        return res_emb



class LearnablePINSenetVggVox256(nn.Module):

    def __init__(self, embedding_dim=4096):

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

    # def forward(self, face, audio):
    def forward(self, face):
        batch_size = face.size(0)
        
        face = self.face_fc(face)
        # audio = self.audio_fc(audio)

        # return face, audio
        return face

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()

    def forward(self, x):
        x = x / torch.norm(x, dim=1, keepdim=True)
        return x

class VGGFace2Model(nn.Module):
    def __init__(self,
                embedding_dim=4096,
                arch_type='senet'
            ):
        super(VGGFace2Model, self).__init__()
        print(arch_type)
        if arch_type=='resnet':
            self.backbone = resnet50()
            # self.backbone.load_state_dict(torch.load('/home/samyak/AudioVisual/ImageFeatureExtractor/resnet.pt'))
        else:
            self.backbone = senet50()
            # self.backbone.load_state_dict(torch.load('/home/samyak/AudioVisual/ImageFeatureExtractor/senet.pt'))

        print("Weights Loaded!")

        for param in self.backbone.parameters():
            param.requires_grad = True

        self.fc = nn.Sequential(
            nn.Linear(2048, embedding_dim),
            nn.ReLU(inplace=True),
            L2Norm()
        )


    def forward(self, x):
        batch_size = x.size(0)
        x = self.backbone(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)

        return x    

if __name__ == '__main__':
    # face senet embeddings
    face_embedding_path = join('data','voxceleb_senet_face_embeddings', 'test')

    # face images
    # face_embedding_path = join('..','VoxCeleb2_dev_test','mp4')

    speaker_embedding_path = join('data','vgg_vox_speaker_embeddings','test')

    model = LearnablePINSenetVggVox256()
    model.load_state_dict(torch.load(join('saved_models','model_e18.pt')))

    # face_model = model.face_fc
    # speech_model = model.audio_fc

    # model = VGGFace2Model()
    # model.load_state_dict(torch.load(join('saved_models','model_e22.pt')))
    # model.load_state_dict(torch.load(join('saved_models','new_loader_senet.pt')))

    face_model = model
    speech_model = 'na'

    embedder = GetEmbeddings(face_model=face_model,speech_model=speech_model)


    acc_arr = []
    for i in range(2, 11):
        evaluation = Evaluation(face_embedding_path=face_embedding_path,speaker_embedding_path=speaker_embedding_path, embedder=embedder,gallery_size=i,num_queries=1000, mode='ff', using_face_embeddings=True)

        acc = evaluation.evaluate()
        acc_arr.append(acc)

    print(acc_arr)
    plt.plot(np.arange(0,len(acc_arr))+2,acc_arr,'r-')
    plt.xlabel('Gallery Size')
    plt.ylabel('Identification Accuracy')
    plt.title('1:N F-V matching')
    plt.grid()
    plt.show()

    


