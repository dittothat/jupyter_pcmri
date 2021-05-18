#!/usr/bin/env python
# coding: utf-8

# Exported from the repo/jupyter/pcmri folder 20201113 and trimed down
# notebook was flow_analysis

# helper functions (getphaseimg_path, getvenc) and flow calculations (calc_flow)

import os
import re 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from nilearn import datasets, plotting, masking
from nilearn.input_data import NiftiSpheresMasker
from nilearn.image import concat_imgs, mean_img, smooth_img, math_img, load_img
from nilearn.plotting import plot_stat_map, plot_anat, plot_img, show
import nilearn.regions
import nibabel as nib
import scipy
import json

# # this will generate the phase image path from the mask fname
def getphaseimg_path(mask_fname, path2data):
    base_filename = mask_fname[:mask_fname.find('mask')-1]
    base_filename = base_filename.replace("gre", "ph")
    return os.path.join(path2data, base_filename + '.nii.gz')

# get vencs
def getvenc(mask_fname, path2data):
#     print('mask_fname=', mask_fname, 'path2data=', path2data)
    path = getphaseimg_path(mask_fname, path2data)
    path = path.replace(".nii.gz", ".json")
    f = open(path, "r")
#     print('path=', path)
    data = json.loads(f.read())
    venc = data['VENC']
    f.close()
    return venc

# This takes a mask file path and phase file path pair and gives you avg vel, area, flow, raw_vel
def calc_flow(mask_path , phase_path, VENC):
    print('\nmask_path:', mask_path)
    mask = load_img(mask_path)
    assert mask.shape[2] == 1, 'Mask is 3D and this will not work'
    phase = load_img(phase_path)
    phase_data = nilearn.masking.apply_mask(phase, mask, dtype='f', smoothing_fwhm=None, ensure_finite=True)
    assert phase.header['pixdim'][1] > 0.1 and phase.header['pixdim'][1] < 1.2, 'Unclear if pixdim is in unit mm'
    # calculate the in-plane area of a voxel
    vox_area = phase.header['pixdim'][1] * phase.header['pixdim'][2] / 100 # cm^2
    print('vox_area = %f' %vox_area)
    # Is the phase for one phase or multiple?
    if phase_data.ndim > 1:
        print('multiple cardiac phases (%i, to be precise) are present!' % phase_data.shape[0])
        # caluclate the area of the mask noting that total voxels in the masked data will be the mask voxels from a single image times phases
        area = phase_data.size * vox_area / phase_data.shape[0]  # cm^2, CURRENTLY THIS FUNCTION WORKS ON A STATIC MASK
        print('area of mask = %f, mask vox count = %i' % (area, phase_data.size / phase_data.shape[0]))
        assert np.max(phase.get_data())-np.min(phase.get_data()) > 8100, 'Wrong phase image scale, not -4096:4096'
        avg_vel = np.mean(phase_data, 1) * VENC / 4096 # images scaled -4096 to 4096, units cm/s
        flow = np.sum(phase_data, 1) * VENC / 4096 * vox_area # ml/s
    else:
        print('one cardiac phase')
        assert phase_data.shape[0] == 1, "Something weird in the data dimensions"
        area = phase_data.size * vox_area  # cm^2
        print('area of mask = %f, mask vox count = %i' % (area, phase_data.size/phase_data.shape[0]))
        assert np.max(phase.get_data())-np.min(phase.get_data()) > 8100, 'Wrong phase image scale, not -4096:4096'
        avg_vel = np.mean(phase_data) * VENC / 4096 # images scaled -4096 to 4096, units cm/s
        flow = np.sum(phase_data) * VENC / 4096 * vox_area # ml/s
    raw_vel = phase_data * VENC / 4096 # cm/s
    return {'fname' : str(phase_path),'avg_vel' : avg_vel, 'area' : area, 'flow' : flow, 'raw_vel': raw_vel}