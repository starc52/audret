import sys
sys.path.append('../')
import argparse
import os
import torch
import time
import torch.nn as nn
import numpy as np
from dataloader import * 
import cv2
import matplotlib.pyplot as plt
from os.path import join
from torchvision import transforms
from utils.utils import *
from tqdm import tqdm
from scipy import spatial
from scipy.io import wavfile
import librosa
from vggm import VGGM
from resnet import *
from senet import *
import traceback
from facenet_pytorch import InceptionResnetV1
from facenet_pytorch import MTCNN
from signal_utils import preprocess
import warnings
warnings.filterwarnings('ignore')


class AudioInference():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)

        # self.model=VGGM(1251)
        # # self.model.load_state_dict(torch.load("../VGGVox-PyTorch-master/models/VGGM300_BEST_140_81.99.pth", map_location=device))
        # self.model.load_state_dict(torch.load("/home/starc52/audret/models/VGGM300_BEST_140_81.99.pth", map_location=self.device))
        # self.model.to(self.device)
        # print("VggVox model loaded")
        # self.model.eval()
        # self.preprocess = preprocess(arch_type=arch_type)
        

    def get_audio_features(self, audio_path, save_path=None, img=None):

        audio,sr=librosa.load(audio_path,sr=16000)

        audio = audio[0:48000]

        audio = preprocess(audio).astype(np.float32)
        audio=np.expand_dims(audio, 2)

        transformers=transforms.ToTensor()
        audio = transformers(audio)
        audio = audio.unsqueeze(0)

        audio = audio.to(self.device)

        with torch.no_grad():
            audio_embedding = self.model(audio)

        audio_embedding = audio_embedding.cpu().numpy()
    
        return audio_embedding

    # todo
    #takes audio as input, splits it with window and stride params, then comptues the embeddings
    def split_audio(self, audio, window=3, stride=2, sr=16000) :
        # window and stride is in seconds

        if type(audio)==str:
            pathToAudio = audio
            audio,sr=librosa.load(audio,sr=16000)

        nsamples_per_window = int(window*sr)
        samples_per_stride = int(stride*sr)

        slice_start = [i for i in range(0,len(audio)-nsamples_per_window,samples_per_stride)]
        slice_end = [i+nsamples_per_window if i+nsamples_per_window<len(audio) else len(audio) for i in slice_start]
        
        audios = [audio[slice_start[i]:slice_end[i]] for i in range(len(slice_start))]
        audios = random.sample(audios, 1)
        audios_processed = [preprocess(audio).astype(np.float32) for audio in audios]
        audios_expanded = [np.expand_dims(audio, 2) for audio in audios_processed]
        try:
            [np.save(join(pathToAudio[:-4]+".npy"), audio_fft) for _, audio_fft in enumerate(audios_expanded)]
            return audios_expanded
        except:
            print("audio_fft directory already present")
        # transformers = transforms.ToTensor()
        # audios_transformed = [transformers(audio) for audio in audios_expanded]
        # audios_unsqzd = [audio.unsqueeze(0) for audio in audios_transformed]
        # embeddings = []
        # try:
        #     os.makedirs(pathToAudio[:-4])
        #     for idx, audio in enumerate(audios_unsqzd):
        #         audio_temp = audio.to(self.device)
        #         with torch.no_grad():
        #             audio_embedding = self.model(audio_temp)
        #         audio_embedding = audio_embedding.cpu().numpy()
        #         np.save(join(pathToAudio[:-4], "%03d.npy"%idx), audio_embedding)
        #         embeddings.append(audio_embedding)
        # except:
        #     print("directory present")
        
        # return embeddings

parser = argparse.ArgumentParser()
parser.add_argument('--root',default='/ssd_scratch/cvit/starc52/VoxCeleb2/dev/mp4', type=str)
parser.add_argument('--model_val_path',default="/home/starc52/Recode/senet.pth", type=str)
parser.add_argument('--start_id', default=0, type=int)
parser.add_argument('--end_id', default=6000, type=int)
args = parser.parse_args()
print(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(post_process=False, keep_all=True, device=device)
AudioModel = AudioInference()

listOfIdentities=sorted(os.listdir(args.root))[args.start_id:args.end_id]
for speaker_id in tqdm(sorted(os.listdir(args.root))[args.start_id:args.end_id]):#just 1000 user ids. 
    for url in sorted(os.listdir(join(args.root, speaker_id))):
        # os.makedirs(join(args.save_path, speaker_id, url), exist_ok=True)
#         for file_name in sorted(os.listdir(join(args.root, speaker_id, url))):
        if not os.path.isdir(join(args.root, speaker_id, url, "frames")):
            continue
        for file_name in sorted(os.listdir(join(args.root, speaker_id, url, "frames"))):
            try:
                if file_name[-4:] == ".jpg":
                    img_file = cv2.imread(join(args.root, speaker_id, url, "frames", file_name), cv2.IMREAD_UNCHANGED)
                    transformToTensor = transforms.ToTensor()                    
                    faces = mtcnn(img_file)
                    if faces is None:
                        count=0
                        print("No face detected for : %s" % join(args.root, speaker_id, url, "frames", file_name))
                        os.remove(join(args.root, speaker_id, url, "frames", file_name))
                        continue
                    img_file = faces[0]
                    img_file = img_file.permute(1, 2, 0).numpy()
                    img_file  = cv2.resize(img_file, (224, 224))
                    cv2.imwrite(join(args.root, speaker_id, url, "frames", file_name[:-4]+".jpg"), img_file)
                    print("Processed filename: %s"% join(args.root, speaker_id, url, "frames", file_name))
            except:
                print(traceback.format_exc())
                continue
for speaker_id in tqdm(sorted(os.listdir(args.root))[args.start_id:args.end_id]):#just 1000 user ids. 
    for url in sorted(os.listdir(join(args.root, speaker_id))):
        if not os.path.isdir(join(args.root, speaker_id, url, "audio")):
            continue
        for file_name in sorted(os.listdir(join(args.root, speaker_id, url, "audio"))):
            try:
                if file_name[-4:] == ".mp3":
                    utteranceEmbedding = AudioModel.split_audio(join(args.root, speaker_id, url, "audio", file_name))
                    os.remove(join(args.root, speaker_id, url, "audio", file_name))
                    print("Processed filename: %s"% join(args.root, speaker_id, url, "audio", file_name))
            except:
                print(traceback.format_exc())
                continue            
print(listOfIdentities)
