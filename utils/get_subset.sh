mkdir -p /ssd_scratch/cvit/samyak/;

rsync -r samyak@ada:/share3/samyak/AudioVisual/subset.zip /ssd_scratch/cvit/samyak/
cd /ssd_scratch/cvit/samyak/
unzip /ssd_scratch/cvit/samyak/subset.zip
cd -
if [ -d /ssd_scratch/cvit/samyak/ ]; then
	rm /ssd_scratch/cvit/samyak/subset.zip -rf
fi
