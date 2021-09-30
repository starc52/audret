import sys
sys.path.append('../')
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import PCA
import os
from os.path import join
import glob
from ImageFeatureExtractor.inference import *


pca = PCA(n_components=3)
pca_kldiv = PCA(n_components=3)
pca_mse = PCA(n_components=3)


path = '/home/samyak/AudioVisual/speaker_embeddings/test_batch'
root_path = '/ssd_scratch/cvit/samyak/mp4/'

model_kldiv = Inference('/home/samyak/AudioVisual/ImageFeatureExtractor/saved_models/new_dataset_kldiv_loss.pt')
model_mse = Inference('/home/samyak/AudioVisual/ImageFeatureExtractor/saved_models/new_dataset_mse_loss.pt')

color = []
embeddings = []
pred_embeddings_kldiv = []
pred_embeddings_mse = []
# for i, speaker_id in enumerate(os.listdir(path)[:10]):
colors = ['r', 'g']
for i, speaker_id in enumerate(['id01460']):
    embed_paths = glob.glob(join(path, speaker_id, '*/*.npy'))
    color.extend([colors[i] for _ in range(len(embed_paths))])
    
    for _path in embed_paths:
        _embed = np.load(_path)
        embeddings.append(_embed)

        lst = _path.split('/')[-3:]
        lst.insert(2, 'frames')
        relative_path = '/'.join(lst)
        relative_path = relative_path.replace('.npy', '.jpg')
        
        _pred_embed_kldiv = model_kldiv.get_signature(join(root_path, relative_path)).reshape(-1)
        pred_embeddings_kldiv.append(_pred_embed_kldiv)

        _pred_embed_mse = model_mse.get_signature(join(root_path, relative_path)).reshape(-1)
        pred_embeddings_mse.append(_pred_embed_mse)

embeddings = np.array(embeddings)
color = np.array(color)
pred_embeddings_kldiv = np.array(pred_embeddings_kldiv)
pred_embeddings_mse = np.array(pred_embeddings_mse)

red_embeddings = pca.fit_transform(embeddings)
red_pred_embeddings_kldiv = pca_kldiv.fit_transform(pred_embeddings_kldiv)
red_pred_embeddings_mse = pca_mse.fit_transform(pred_embeddings_mse)

ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=red_embeddings[:,0], 
    ys=red_embeddings[:,1], 
    zs=red_embeddings[:,2], 
    c=color,
    marker='.'
)

ax.scatter(
    xs=red_pred_embeddings_kldiv[:,0], 
    ys=red_pred_embeddings_kldiv[:,1], 
    zs=red_pred_embeddings_kldiv[:,2], 
    c=color,
    marker='<'
)

ax.scatter(
    xs=red_pred_embeddings_mse[:,0], 
    ys=red_pred_embeddings_mse[:,1], 
    zs=red_pred_embeddings_mse[:,2], 
    c=color,
    marker='^'
)

ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')

plt.savefig('plot_embed_one.jpg')