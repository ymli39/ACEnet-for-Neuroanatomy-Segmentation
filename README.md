# ACEnet-for-Neuroanatomy-Segmentation
ACEnet: Anatomical Context-Encoding Network for Neuroanatomy Segmentation


-------------------------------------------------------------

## Preprocessing
Data can be found at 2012 MALC MICCAI challenge website.
1. Use Freesurfer to preprocess data into 256x256x256 along with the brainmask generated.
2. convert data and brainmasks to numpy, refer to ./data_utils/utils.py function remaplabels() to generate corresponding labels.

The file should follow the directory:

    |-->Project
        |-->resampled
           |-->training-imagesnpy (1000_3.npy and 1000_3_brainmask.npy)
           |-->training-labels-remapnpy (1000_3_glm.npy --labels for coarse segmentation)
           |-->training-labels139 (1000_3_glm.npy --labels for fine-grained segmentation)
           |-->testing-imagesnpy (1003_3.npy and 1000_3_brainmask.npy)
           |-->testing-labels-remapnpy (1003_3_glm.npy --labels for coarse segmentation)
           |-->testing-labels139 (1003_3_glm.npy --labels for fine-grained segmentation)
        |-->segmentation
           |-->all git files

## Installation:
```
git clone https://github.com/ymli39/ACEnet-for-Neuroanatomy-Segmentation
cd ACEnet-for-Neuroanatomy-Segmentation
pip install nibabel tqdm
```


-------------------------------------------------------------

## Training
Parameter could be tuned at the beginning of the running files: train.py, test_coarse.py, test_fine.py.

You need to modify the folowing subjects for training and testing:
```
RESUME_PATH: directory to resume the model
SAVE_DIR: directory to save the model
NUM_CLASS: label classes +1 (background)
TWO_STAGES: use two stage training
RESUME_PRETRAIN: set False if want to train from epoch 0, True to resume the pretrained epoch

DATA_DIR = '../resampled/'
DATA_LIST = './datasets/'

-b-train: For NVIDIA TITAN XP GPU with 12 GB memory, use batch size of 4. 
-b-test: use 2, must be bigger than 1.
-num-slices: slice thickness used for Spatial Encoding Module, use 3 for coase-grained segmentation and 7 for for-grained segmentation.
--lr-scheduler: used poly
--lr: for train from scratch, use 0.01 and 0.02 for coarse and fine-grained respecitvely, for pretrain, use 0.001 and 0.005 for coarse and fine-grained respecitvely.
```

For start a new training, use:
```
CUDA_VISIBLE_DEVICES=0 python train.py --resume-pretrain False
```

For load the data augmented pretrain model, use:
```
CUDA_VISIBLE_DEVICES=0 python train.py --resume-pretrain True
```

For running the test, use:
```
CUDA_VISIBLE_DEVICES=0 python test_(coarse/fine).py
```

-------------------------------------------------------------

## Testing
I have updated a test_demo folder for people to use, this folder contains the models trained on 30 MALC 2012 dataset in both coarse-grained and fine-grained segmentations. 

You could chose any MRI images to generate corresponding segmentation labels. This model takes the input of a MRI brain images and outputs the setgmentation mask and skull mask.

The testing run script is referred in file "runscript.txt"

-------------------------------------------------------------

## Update

I added a dataloader file for loading nifty data. The file could be found under directory: ./data_utils/MRIloader_nifty.py

-------------------------------------------------------------

## Reference
Please refer to the paper for more implementation details:
```
@article{li2021acenet,
  title={ACEnet: Anatomical Context-Encoding Network for Neuroanatomy Segmentation},
  author={Li, Yuemeng and Li, Hongming and Fan, Yong},
  journal={Medical Image Analysis},
  pages={101991},
  year={2021},
  publisher={Elsevier}
}
```
