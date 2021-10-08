import numpy as np
import os
import shutil
import sys
import traceback
from tqdm import tqdm

root = "/ssd_scratch/cvit/starc52/VoxCeleb2/test/mp4/"
count=0
for _id in tqdm(os.listdir(root)):
    for _url in os.listdir(os.path.join(root, _id)):
        listOfFolders = [f for f in os.listdir(os.path.join(root, _id, _url)) if os.path.isdir(os.path.join(root, _id, _url, f))]
        for numb in listOfFolders:
            for sample in os.listdir(os.path.join(root, _id, _url, numb)):
                try:
                    a=np.load(os.path.join(root, _id, _url, numb, sample))
                except:
                    #print(traceback.format_exc())
                    count+=1
                    #print(os.path.join(root, _id, _url, numb, sample))
                    os.remove(os.path.join(root, _id, _url, numb, sample))
                    #sys.exit()
print(count)