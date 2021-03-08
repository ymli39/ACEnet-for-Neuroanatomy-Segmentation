#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yuemeng Li
"""

'''
ignored labels reference:
https://gitlab.icm-institute.org/aramislab/clinica/blob/ea014f490183a5a4c0b834db38677a6af650a153/clinica/resources/atlases_spm/Neuromorphometrics.txt
'''
import numpy as np
import nibabel as nib
import os
import csv
import pdb
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool


def estimate_weights_mfb(labels):
    labels = labels.astype(np.float64)
    class_weights = np.zeros_like(labels)
    unique, counts = np.unique(labels, return_counts=True)
    median_freq = np.median(counts)
    weights = np.zeros(len(unique))
    for i, label in enumerate(unique):
        class_weights += (median_freq // counts[i]) * np.array(labels == label)
        weights[int(label)] = median_freq // counts[i]

    grads = np.gradient(labels)
    edge_weights = (grads[0] ** 2 + grads[1] ** 2) > 0
    class_weights += 2 * edge_weights
    return class_weights, weights

def remaplabels(id, labelfiles, labeldir, savedir):
    
    labelfile = labelfiles[id]
    
    if os.path.exists(os.path.join(savedir,labelfile[:-7]+'.npy')):
        return
        
    label = nib.load(os.path.join(labeldir,labelfile))
    labelnpy = label.get_fdata()
    labelnpy = labelnpy.astype(np.int32)
#        labelnpy = np.load(os.path.join(labeldir,labelfile))
    
    
#        labelnpy[(labelnpy >= 100) & (labelnpy % 2 == 0)] = 210
#        labelnpy[(labelnpy >= 100) & (labelnpy % 2 == 1)] = 211
#        label_list = [45, 211, 44, 210, 52, 41, 39, 60, 37, 58, 56, 4, 11, 35, 48, 32, 62, 51, 40, 38, 59, 36, 57, 
#                      55, 47, 31, 61]
    
    label_list = np.array([4,  11,  23,  30,  31,  32,  35,  36,  37,  38,  39,  40,
                    41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  55,
                    56,  57,  58,  59,  60,  61,  62,  63,  64,  69,  71,  72,  73,
                    75,  76, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 112,
                   113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                   128, 129, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
                   143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
                   156, 157, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
                   171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
                   184, 185, 186, 187, 190, 191, 192, 193, 194, 195, 196, 197, 198,
                   199, 200, 201, 202, 203, 204, 205, 206, 207])

    new_labels = np.zeros_like(labelnpy)

    for i, num in enumerate(label_list):
        label_present = np.zeros_like(labelnpy)
        label_present[labelnpy == num] = 1
        new_labels = new_labels + (i + 1) * label_present
    
#        checkind = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27])
    if (np.sum(np.unique(new_labels)>139) and np.sum(np.unique(new_labels)<0)) > 0:
        print('error')
    
    np.save(os.path.join(savedir,labelfile[:-7]), new_labels)
    print('finished converting label: ' + labelfile)
#        img = nib.Nifti1Image(new_labels, np.eye(4))
#        img.get_data_dtype() == np.int32
#        nib.save(img, os.path.join(savedir,labelfile))
    

def convertTonpy(datadir, savedir):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    datafiles = [f for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, f))]
    datafiles = [f for f in datafiles if '_img.nii.gz' in f]
#    niftyfile = [f for f in datafiles if '.nii.gz' in f]
#    nifty1_5T = [f for f in niftyfile if '1.5T' in f and 'deformed' not in f]
#    nifty3_0T = [f for f in niftyfile if '3.0T' in f]
#    nifty1_5Twarp = [f for f in niftyfile if '1.5T' in f and 'deformed' in f]
#    datafiles = nifty3_0T + nifty1_5Twarp
    
    tbar = tqdm(datafiles)
    
    for datafile in tbar:
        data = nib.load(os.path.join(datadir,datafile))
        datanpy = data.get_fdata()
#        datanpy = datanpy.astype(np.int32)
        datanpy = datanpy.astype(np.float32)
        np.save(os.path.join(savedir,datafile[:-7]), datanpy)


def convertToNifty(datadir, savedir):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
        
    datafiles = [f for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, f))]
    
    for datafile in datafiles:
        datanpy = np.load(os.path.join(datadir,datafile))
        datanpy = np.transpose(datanpy, (1,2,0))
        datanpy = datanpy.astype(np.uint8)
        img = nib.Nifti1Image(datanpy, np.eye(4))
        assert(img.get_data_dtype() == np.uint8)
        nib.save(img, os.path.join(savedir,datafile[:-4]+'.nii.gz'))
        

def process_fine_labels(fine_label_dir):
    dice_score = np.load(os.path.join(fine_label_dir, 'dice_score.npy'))
    iou_score = np.load(os.path.join(fine_label_dir, 'iou_score.npy'))
    
    label_list = np.array([4,  11,  23,  30,  31,  32,  35,  36,  37,  38,  39,  40,
                       41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  55,
                       56,  57,  58,  59,  60,  61,  62,  63,  64,  69,  71,  72,  73,
                       75,  76, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 112,
                      113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                      128, 129, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
                      143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
                      156, 157, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
                      171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
                      184, 185, 186, 187, 190, 191, 192, 193, 194, 195, 196, 197, 198,
                      199, 200, 201, 202, 203, 204, 205, 206, 207])
    total_idx = np.arange(0, len(label_list))
    ignore = np.array([42, 43, 64, 69])
    
    valid_idx = [i+1 for i in total_idx if label_list[i] not in ignore]
    valid_idx = [0] + valid_idx
    
    dice_score_vali = dice_score[:,valid_idx]
    iou_score_vali = iou_score[:,valid_idx]
    
    np.mean(dice_score_vali)
    np.std(dice_score_vali)
    
    np.mean(iou_score_vali)
    np.std(iou_score_vali)
    
    
if __name__ == '__main__':
    print('This is utils')
    
    
    
    
    
    
