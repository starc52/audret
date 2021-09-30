import os
from os.path import join

train_path = '/ssd_scratch/cvit/samyak/VoxCeleb2_dev_test/dev/mp4'
subset_train_path = '/ssd_scratch/cvit/samyak/VoxCeleb2_dev_test/subset/dev/mp4'

test_path = '/ssd_scratch/cvit/samyak/VoxCeleb2_dev_test/mp4'
subset_test_path = '/ssd_scratch/cvit/samyak/VoxCeleb2_dev_test/subset/mp4'

os.makedirs(subset_train_path, exist_ok=True)
os.makedirs(subset_test_path, exist_ok=True)

for (i, j, num) in [(train_path, subset_train_path, 100), (test_path, subset_test_path, 10)]:
	lst = os.listdir(i)[:num]
	for k in lst:
		os.system('cp -r {} {}'.format(join(i, k), j))