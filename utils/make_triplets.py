import os
from os.path import join
from glob import glob
import random
import json

train_root = '/ssd_scratch/cvit/samyak/VoxCeleb2/dev/mp4'
test_root = '/ssd_scratch/cvit/samyak/VoxCeleb2/mp4'

embedding_train_path = '../speaker_embeddings/train_batch'
embedding_test_path = '../speaker_embeddings/test_batch'



for (path, embed_path, save_file) in [(train_root, embedding_train_path, 'train_triplets.json'), (test_root, embedding_test_path, 'test_triplets.json')]:
	speaker_ids = sorted(os.listdir(embed_path))

	final_arr = []
	for i, _id in enumerate(speaker_ids):
		pos_samples = glob(join(embed_path, _id, '*/*.npy'))

		for pos_path in pos_samples:
			rand_idx = i
			while rand_idx==i:
				rand_idx = random.randint(0, len(speaker_ids)-1)

			neg_samples = glob(join(embed_path, speaker_ids[rand_idx], '*/*.npy'))
			neg_path = neg_samples[random.randint(0, len(neg_samples)-1)]

			[url, file_id] = pos_path.split('/')[-2:]

			final_arr.append({
				'img_path': join(path, _id, url, 'frames',file_id.replace('.npy', '.jpg')),
				'positive_path': pos_path,
				'negative_path': neg_path
			})
	print(len(final_arr))
	with open(save_file, 'w') as f:
		json.dump(final_arr, f, sort_keys=True, indent=4)