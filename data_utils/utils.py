##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Created by: Yuemeng Li
#BE department, University of Pennsylvania
#Email: ymli@seas.upenn.edu
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import nibabel as nib
import os
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.spatial.distance import directed_hausdorff
import data_utils.surface_distance as surface_distance


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
    
    ########################this is for coarse-grained dataset##############################
#        labelnpy[(labelnpy >= 100) & (labelnpy % 2 == 0)] = 210
#        labelnpy[(labelnpy >= 100) & (labelnpy % 2 == 1)] = 211
#        label_list = [45, 211, 44, 210, 52, 41, 39, 60, 37, 58, 56, 4, 11, 35, 48, 32, 62, 51, 40, 38, 59, 36, 57, 
#                      55, 47, 31, 61]
    
    ########################this is for fine-grained dataset##############################
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
    
    if (np.sum(np.unique(new_labels)>139) and np.sum(np.unique(new_labels)<0)) > 0:
        print('error')
    
    np.save(os.path.join(savedir,labelfile[:-7]), new_labels)
    print('finished converting label: ' + labelfile)
    
    
def process_resampled_labels():
    labeldir = None #label directory for all nifty images
    savedir = None #directory to save numpy images
    
    labelfiles = [f for f in os.listdir(labeldir) if os.path.isfile(os.path.join(labeldir, f))]
    labelfiles = [f for f in labelfiles if '_lab.nii.gz' in f]

    pool = Pool(processes=20)
    partial_mri = partial(remaplabels, labelfiles=labelfiles, labeldir=labeldir, savedir=savedir)
    
    pool.map(partial_mri, range(len(labelfiles)))
    pool.close()
    pool.join()
    print('end preprocessing brain data')
    
    
def convertTonpy(datadir, savedir):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    datafiles = [f for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, f))]
    datafiles = [f for f in datafiles if '_brainmask.nii.gz' in f]
    
    tbar = tqdm(datafiles)
    
    for datafile in tbar:
        data = nib.load(os.path.join(datadir,datafile))
        datanpy = data.get_fdata()
        datanpy = datanpy.astype(np.int32)
#        datanpy = datanpy.astype(np.float32)
        datanpy[datanpy>0]=1
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
    
    print(np.mean(dice_score_vali))
    print(np.std(dice_score_vali))
    
    print(np.mean(iou_score_vali))
    print(np.std(iou_score_vali))
    
    
def remap_IXI_images(id, subfodlers, savedir):

    subname = subfodlers[id]
    
    orig_dir = subname+'_orig.nii.gz'
    aseg_dir = subname+'_aseg.nii.gz'
    brain_mask_dir = subname+'_brainmask.nii.gz'
    name = subname.split('/')[-1]
    
    
    orig = nib.load(orig_dir)
    orig_npy = orig.get_fdata()
    orig_npy = orig_npy.astype(np.int32)
    np.save(os.path.join(savedir,'training_images/'+name+'.npy'), orig_npy)
    
    aseg = nib.load(aseg_dir)
    aseg_npy = aseg.get_fdata()
    aseg_npy = aseg_npy.astype(np.int32)
    correspond_labels = [2, 3, 41, 42, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 28, 43, 46, 47, 
                         49, 50, 51, 52, 53, 54, 60]
    
    new_labels = np.zeros_like(aseg_npy)

    for i, num in enumerate(correspond_labels):
        label_present = np.zeros_like(aseg_npy)
        label_present[aseg_npy == num] = 1
        new_labels = new_labels + (i + 1) * label_present
    np.save(os.path.join(savedir,'training_labels/'+name+'.npy'), new_labels)
    
    brain_mask = nib.load(brain_mask_dir)
    brain_mask_npy = brain_mask.get_fdata()
    brain_mask_npy = brain_mask_npy.astype(np.int32)
    brain_mask_npy[brain_mask_npy>0]=1
    np.save(os.path.join(savedir,'training_skulls/'+name+'.npy'), brain_mask_npy)
        
    print('finished processing image '+name)
        
def process_IXI_images():
    datadir = '/IXI_T1_surf/'
    nii_path = '/IXI_T1_surf_nii/'
    savedir = '/IXI_T1_surf/'
    
    subfodlers = [os.path.join(nii_path, name) for name in os.listdir(datadir) if os.path.isdir(os.path.join(datadir, name)) and 'IXI' in name]

    pool = Pool(processes=20)
    partial_mri = partial(remap_IXI_images, subfodlers=subfodlers, savedir=savedir)
    
    pool.map(partial_mri, range(len(subfodlers)))
    pool.close()
    pool.join()
    print('end preprocessing IXI data')
    

def compute_Hausdorff_distance(id, subfiles, gt_dir, pred_dir, baseline_dir):
    file_name = subfiles[id]
    
#    subfiles = [name for name in os.listdir(gt_dir)]
    
    dist_pred_lists = []
    dist_quick_lists = []
    
#    label_list = np.array([4,  11,  23,  30,  31,  32,  35,  36,  37,  38,  39,  40,
#                       41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  55,
#                       56,  57,  58,  59,  60,  61,  62,  63,  64,  69,  71,  72,  73,
#                       75,  76, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 112,
#                      113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
#                      128, 129, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
#                      143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
#                      156, 157, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
#                      171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
#                      184, 185, 186, 187, 190, 191, 192, 193, 194, 195, 196, 197, 198,
#                      199, 200, 201, 202, 203, 204, 205, 206, 207])
#    total_idx = np.arange(0, len(label_list))
#    
#    ignore = np.array([42,43,64, 69])
#    
#    non_valid_idx = [i+1 for i in total_idx if label_list[i] in ignore]
#    non_valid_idx = non_valid_idx
        
    #this is originally a for loop
    file_gt = nib.load(os.path.join(gt_dir,file_name))
    file_gt_npy = file_gt.get_fdata().astype(np.int32)
    
    file_pred = nib.load(os.path.join(pred_dir,file_name))
    file_pred_npy = file_pred.get_fdata().astype(np.int32)
    
    file_quick = nib.load(os.path.join(baseline_dir,file_name))
    file_quick_npy = file_quick.get_fdata().astype(np.int32)
    
#    for idx in non_valid_idx:
#        file_pred_npy[file_pred_npy==idx] = 0
#        file_quick_npy[file_quick_npy==idx] = 0
#        file_gt_npy[file_gt_npy==idx] = 0
    
    for idx in np.unique(file_gt_npy):
        temp_gt = np.zeros_like(file_gt_npy)
        temp_pred = np.zeros_like(file_gt_npy)
        temp_quick = np.zeros_like(file_gt_npy)
        
        temp_gt[file_gt_npy==idx] = 1
        temp_pred[file_pred_npy==idx] = 1
        temp_quick[file_quick_npy==idx] = 1
        
        surface_distances_pred = surface_distance.compute_surface_distances(temp_gt, temp_pred, spacing_mm=(1, 1, 1))
        dist_pred = surface_distance.compute_robust_hausdorff(surface_distances_pred, 100)
        dist_pred_lists.append(dist_pred)
        
        surface_distances_quick = surface_distance.compute_surface_distances(temp_gt, temp_quick, spacing_mm=(1, 1, 1))
        dist_quick = surface_distance.compute_robust_hausdorff(surface_distances_quick, 100)
        dist_quick_lists.append(dist_quick)
        
    np.save(os.path.join(pred_dir,file_name+'_pred.npy'), dist_pred_lists)
    np.save(os.path.join(pred_dir,file_name+'_quick.npy'), dist_quick_lists)
            

def process_Hausdorff_distance():
    baseline_dir = '/MRI_model/quicknat/pred' #this is our baseline
    pred_dir = '/MRI_model/coarse_dir/pred'
    gt_dir = None #gt label directory, double check the rotation matches with pred
    

    subfiles = [name for name in os.listdir(gt_dir)]

    pool = Pool(processes=16)
    partial_mri = partial(compute_Hausdorff_distance, subfiles=subfiles, gt_dir=gt_dir, pred_dir=pred_dir, baseline_dir=baseline_dir)
    
    pool.map(partial_mri, range(len(subfiles)))
    pool.close()
    pool.join()
    print('end preprocessing IXI data')
    

def process_Hausdorff_npy():
    #this is for MALC27
    gt_dir = '/label/' #gt label directory, double check the rotation matches with pred
    pred_dir = '/MRI_model/coarse_dir/pred'
    
    subfiles = [name for name in os.listdir(gt_dir)]
    
    dist_pred_lists = []
    dist_quick_lists = []
    for file_name in subfiles:
        pred_dist = np.load(os.path.join(pred_dir,file_name+'_pred.npy'))
        quick_dist = np.load(os.path.join(pred_dir,file_name+'_quick.npy'))
        quick_dist[quick_dist==np.inf]=150
        dist_pred_lists.append(np.mean(pred_dist[1:,0].astype(np.float32)))
        dist_quick_lists.append(np.mean(quick_dist[1:,0].astype(np.float32)))
    dist_pred_lists = np.asarray(dist_pred_lists)
    dist_quick_lists = np.asarray(dist_quick_lists)
    print(np.mean(dist_pred_lists))
    print(np.mean(dist_quick_lists))
    
    
if __name__ == '__main__':
    process_resampled_labels()
    
