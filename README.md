# audret
Procedure to use this code:
* copy the VoxCeleb2 dataset from share3/
* check all the paths in all files to be appropriate to the location of the dataset.
* run ```preprocess.py``` in batches on the code, as it is quite slow to run on the entire VoxCeleb2 dataset.
* The output of ```preprocess.py``` is face images cropped from the videos using mtcnn, and preprocessed form of audio as ```.npy``` for easy dataloading.
* then run ```retrieveTrain.py``` for training the models. The current model loads both branches' pretrained model and finetunes all parameters. you can change that as you wish. 
* ```retrieveTrain.py``` will produce models and save it for the number of epochs it is run for. 
* run ```evaluate.py``` for evaluating the dataset for the verification task. 
* run ```evaluate_lossy.py``` for evaluation of the dataset in a reduced resolution scenario. 
* Both evaluation scripts produce matplotlib plots providing the accuracy in various verification scenarios.  
