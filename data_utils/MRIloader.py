##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Created by: Yuemeng Li
#BE department, University of Pennsylvania
#Email: ymli@seas.upenn.edu
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


import torch
import os
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from data_utils.utils import estimate_weights_mfb


class LoadMRIData(Dataset):
    
    def __init__(self, mri_dir, list_dir, phase, num_class, num_slices=3, se_loss = True, Encode3D = False, use_weight = False):
        "load MRI into a 2D slice and a 3D image"
        
        self.phase = phase
        self.se_loss = se_loss
        self.Encode3D = Encode3D
        self.num_class = num_class
        self.use_weight = use_weight
        self.num_slices = num_slices
        
        if self.use_weight:
            weight_dir = os.path.join(mri_dir, 'training-weightsnpy')
            self.weight_names = []
        
        if self.phase is 'train':
            data_dir = os.path.join(mri_dir, 'training-imagesnpy')
            if num_class is 28:
                label_dir = os.path.join(mri_dir, 'training-labels-remapnpy')
            else:
                label_dir = os.path.join(mri_dir, 'training-labels139')
            image_list = os.path.join(list_dir, 'train_volumes.txt')
            
            self.image_names = []
            self.image_slices = []
            self.label_names = []
            self.skull_names = []
            with open(image_list, 'r') as f:
                for line in f:
                    for i in range(256):
                        image_name = os.path.join(data_dir, line.rstrip() + '.npy')
                        label_name = os.path.join(label_dir, line.rstrip() + '_glm.npy')
                        skull_name = os.path.join(data_dir, line.rstrip() + '_brainmask.npy')
                        self.image_names.append(image_name)
                        self.label_names.append(label_name)
                        self.skull_names.append(skull_name)
                        self.image_slices.append(i)
                        
                        if self.use_weight:
                            weight_name = os.path.join(weight_dir, line.rstrip() + '_glm.npy')
                            self.weight_names.append(weight_name)
        elif self.phase is 'test':
            data_dir = os.path.join(mri_dir, 'testing-imagesnpy')
            if num_class is 28:
                label_dir = os.path.join(mri_dir, 'testing-labels-remapnpy')
            else:
                label_dir = os.path.join(mri_dir, 'testing-labels139')
            image_list = os.path.join(list_dir, 'test_volumes.txt')
            
            self.image_names = []
            self.image_slices = []
            self.label_names = []
            with open(image_list, 'r') as f:
                for line in f:
                    image_name = os.path.join(data_dir, line.rstrip() + '.npy')
                    label_name = os.path.join(label_dir, line.rstrip() + '_glm.npy')
                    self.image_names.append(image_name)
                    self.label_names.append(label_name)

    
    def __getitem__(self, idx):
        #this is for non-pre-processing data        
        image_name = self.image_names[idx]
        label_name = self.label_names[idx]
        
        img_3D = np.load(image_name)
        #normalize data
        img_3D = (img_3D.astype(np.float32)-128)/128
        label_3D = np.load(label_name).astype(np.int32)
        
        if self.phase is 'train':
            x_ax, y_ax, z_ax = np.shape(img_3D)
            skull_name = self.skull_names[idx]
            skull_3D = np.load(skull_name).astype(np.int32)
            
            image_slice = self.image_slices[idx]
            img_coronal = img_3D[:,:,image_slice][np.newaxis,:,:]
            label_coronal = label_3D[:,:,image_slice]
            skull_coronal = skull_3D[:,:,image_slice]
            
            sample = {'image': torch.from_numpy(img_coronal), 'label': torch.from_numpy(label_coronal),
                      'skull': torch.from_numpy(skull_coronal)}
        
            if self.se_loss:
                curlabel = np.unique(label_coronal)
                cls_logits = np.zeros(self.num_class, dtype = np.float32)
                if np.sum(curlabel > self.num_class) >0:
                    curlabel[curlabel>self.num_class] = 0
                cls_logits[curlabel] = 1
                sample['se_gt'] = torch.from_numpy(cls_logits)
            
            if self.Encode3D:
                if image_slice<=int(self.num_slices*2+1):
                    image_stack = img_3D[:,:,0:int(self.num_slices*2+1)]
                elif image_slice >=z_ax-int(self.num_slices*2+1):
                    image_stack = img_3D[:,:,z_ax-int(self.num_slices*2+1):]
                else:
                    image_stack = img_3D[:,:,image_slice-self.num_slices:image_slice+self.num_slices+1]
                image_stack = np.transpose(image_stack, (2,0,1))
                sample['image_stack'] = torch.from_numpy(image_stack)
                
            #estimate class weights
            if self.use_weight:
                weight_name = self.weight_names[idx]
                weights_3D = np.load(weight_name).astype(np.float32)
                weight_slice = weights_3D[:,:,image_slice]
                sample['weights'] = torch.from_numpy(weight_slice)
            
        if self.phase is 'test':
            img_3D = np.transpose(img_3D, (2,0,1))
            label_3D = np.transpose(label_3D, (2,0,1))
            name = image_name.split('/')[-1][:-4]
            sample = {'image_3D': torch.from_numpy(img_3D), 'label_3D': torch.from_numpy(label_3D),
                      'name': name}
            
        return sample
        
    
    def __len__(self):
        return len(self.image_names)
        

if __name__ == '__main__':
    data_dir = '/home/ym/Desktop/research/MRI_seg/resampled'
    image_list = '/home/ym/Desktop/research/quickNAT_pytorch-master/datasets'
    
    traindata = LoadMRIData(data_dir, image_list, 'train')
    train_loader = DataLoader(traindata, batch_size = 1, shuffle = True, num_workers = 8, pin_memory=True)
    
    it = iter(train_loader)
    first = next(it)
    image = first[0]
    label = first[1]
    