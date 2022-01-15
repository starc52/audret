import os
from os.path import join

train_path = '/ssd_scratch/cvit/samyak/VoxCeleb2_dev_test/dev/mp4'
test_path = '/ssd_scratch/cvit/samyak/VoxCeleb2_dev_test/mp4'

for path in [train_path, test_path]:
	for utterance in sorted(os.listdir(path)):
		for link in sorted(os.listdir(join(path, utterance))):
			for video in sorted(os.listdir(join(path, utterance, link))):
				if video.endswith(".mp4"):
					os.makedirs(join(path, utterance, link, 'frames'), exist_ok=True)
					os.makedirs(join(path, utterance, link, 'audio'), exist_ok=True)

					os.system("ffmpeg -i {} -vframes 1 {}".format(
							join(path, utterance, link, video),
							join(path, utterance, link, 'frames', video.split(".")[0]+".jpg")
						))

					os.system("ffmpeg -i {} -q:a 0 -map a {}".format(
							join(path, utterance, link, video),
							join(path, utterance, link, 'audio', video.split(".")[0]+".mp3")
						))

					