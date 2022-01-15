a=$USER
mkdir -p /ssd_scratch/cvit/$a/VoxCeleb2
rsync -r $a@ada:/share3/samyak/VoxCeleb2_AudioImage.zip /ssd_scratch/cvit/$a/VoxCeleb2/
cd /ssd_scratch/cvit/$a/VoxCeleb2
echo "Copied Dataset"
unzip -qq VoxCeleb2_AudioImage.zip

cd -

if [ -d /ssd_scratch/cvit/$a/VoxCeleb2/dev ]; then
	rm /ssd_scratch/cvit/$a/VoxCeleb2/VoxCeleb2_AudioImage.zip -rf
fi
