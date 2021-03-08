#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yuemeng Li
"""

from backbone import Backbone
import torch
import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import DataParallel
import nibabel as nib
import pdb


#######run command###########
#python CUDA_VISIBLE_DEVICES=0 test.py --data-name /home/yli/MRI_project/resampled/testing-images/1003_3.nii.gz --save-dir ./MRI_model/MALC_coarse/output --seg-task coarse

class Solver():
    def __init__(self, args, mri_data, num_class):
        self.args = args
        self.num_class = num_class
        self.mri_data = mri_data
        
        model = Backbone(num_class, args.num_slices)
        
        self.model = model
        
         # Using cuda
        if args.cuda:
            self.model = DataParallel(self.model).cuda()
        
        # Resuming checkpoint
        if not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
        checkpoint = torch.load(args.resume)
        
        if args.cuda:
            self.model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(args.resume))

        
    def validation(self, epoch):
        self.model.eval()
        batch_size = self.args.test_batch_size
        
        with torch.no_grad():
            
            volume = self.mri_data.type(torch.FloatTensor)
            volume = torch.squeeze(volume)
            
            if self.args.cuda:
                volume = volume.cuda()
            
            z_ax, x_ax, y_ax = np.shape(volume)
            
            volume_prediction = []
            skull_prediction = []
            for i in range(0, len(volume), batch_size):
                if i<=int(self.args.num_slices*2+1):
                    image_stack0 = volume[0:int(self.args.num_slices*2+1),:,:][None]
                    image_stack1 = volume[1:int(self.args.num_slices*2+2),:,:][None]
                elif i >=z_ax-int(self.args.num_slices*2+1):
                    image_stack0 = volume[z_ax-int(self.args.num_slices*2+2):-1,:,:][None]
                    image_stack1 = volume[z_ax-int(self.args.num_slices*2+1):,:,:][None]
                else:
                    image_stack0 = volume[i-self.args.num_slices:i+self.args.num_slices+1,:,:][None]
                    image_stack1 = volume[i-self.args.num_slices+1:i+self.args.num_slices+2,:,:][None]
                
                image_3D = torch.cat((image_stack0, image_stack1), dim =0)
                
                outputs = self.model(image_3D)
                pred = outputs[0]
                skull = outputs[2]
                
                _, batch_output = torch.max(pred, dim=1)
                volume_prediction.append(batch_output)
                
                _, skull_output = torch.max(skull, dim=1)
                skull_prediction.append(skull_output)
            
            #volume and label are both CxHxW
            volume_prediction = torch.cat(volume_prediction)
            skull_prediction = torch.cat(skull_prediction)
            print('finished generating mask and skull')
            
        return volume_prediction, skull_prediction
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                        help='number of data loading workers (dcriterionefault: 8)')
    parser.add_argument('--save-dir', default='', type=str, metavar='SAVE',
                        help='directory to save checkpoint (default: none)')
    parser.add_argument('--data-name', type=str, default='',
                        help='directory to load data')
    parser.add_argument('--seg-task', default='', type=str,
                        help='implement coarse or fine grained segmentation')
    # training hyper params
    parser.add_argument('-b-test', '--test-batch-size', default=2, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('-num-slices', '--num-slices', default=5, type=int,
                        metavar='N', help='slice thickness for spatial encoding')
    # cuda, seed and loggingevaluator
    parser.add_argument('--cuda', action='store_false', default=True,
                        help='check whether to use cuda')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    
    if args.seg_task == 'coarse':
        print('start coarse grained segmentation with 27 labels')
        num_class = 28
        args.resume = './MRI_model/MALC_coarse/checkpoint.pth.tar'
    elif args.seg_task == 'fine':
        print('start fine grained segmentation with 138 labels')
        num_class = 139
        args.resume = './MRI_model/MALC_fine/checkpoint.pth.tar'
    
    
    '''1. load data'''
    data = nib.load(args.data_name)
    mri_data = data.get_fdata()
    
    
    #normalize data
    mri_data = (mri_data.astype(np.float32)-128)/128
    mri_data = np.transpose(mri_data, (2,0,1))
    mri_data = torch.from_numpy(mri_data)
    
    
    '''2. load dl model'''
    print(args)
    solver = Solver(args, mri_data, num_class)
    
    
    '''3. Running dl model'''
    print('Running model...')
    volume_prediction, skull_prediction = solver.validation(0)
    
    
    '''4. save seg and skull images''' 
    #########################save output to directory##################################
    volume_prediction = volume_prediction.cpu().numpy().astype(np.uint8)
    volume_prediction = np.transpose(volume_prediction, (1,2,0))
    nib_pred = nib.Nifti1Image(volume_prediction, affine=np.eye(4))
    nib.save(nib_pred, args.save_dir + '_pred.nii.gz')
    
    skull_prediction = skull_prediction.cpu().numpy().astype(np.uint8)
    skull_prediction = np.transpose(skull_prediction, (1,2,0))
    nib_skull = nib.Nifti1Image(skull_prediction, affine=np.eye(4))
    nib.save(nib_skull, args.save_dir + '_skull.nii.gz')
    #########################save output to directory##################################
    print('Saving output images...')
    
if __name__ == '__main__':
    main()
    
    
