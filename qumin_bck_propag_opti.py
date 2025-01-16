# QUMIN PROJECT
# Backpropagation tests


## Imports

import time
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread, imwrite
import sys
import os
# local file
sys.path.insert(0, os.path.abspath("/home/fy106182/Documents/ATER/recherche/QUMIN/python_qumin"))
from qumin_lib import *


## directory and path
# initial file
main_dir = '/home/fy106182/Documents/ATER/recherche/QUMIN/20240711_sample_4/test_cell_z_stack/'
file_name = 'test_cell_z_stack_MMStack_Pos0.ome.tif'

# new files to be created
normalized_name = "z_stack.tiff"
intensity_name = "bckprg_I.tiff"
phase_name = "bckprg_phi.tiff"

# creating their path
file_path = os.path.join(main_dir, file_name)
intensity_path = os.path.join(main_dir, intensity_name)
phase_path = os.path.join(main_dir, phase_name)
normalized_path = os.path.join(main_dir, normalized_name)

## setting parameters and slices
background_ranges = (slice(100, 900), slice(50, 1200))
image_ranges = (slice(706, 706+1024), slice(1703, 1703+1024))
ref_image_num = 13 # image that is going to be used for backpropagation

z = (28 + np.arange(-5, 5.25, 0.25)) * 1e-6 # z values

lambda_ = 638e-9 # wavelength (m)
pp = 0.0993442e-6  # Pixel size in m/pix
n = 1.333 # refractive index
zero_padd = 0

show = True # for the initial plot
w = 0  # For different warnings


## reading files
im = np.array(imread(file_path), dtype=float)

# get median
bckgrd = im[:,background_ranges[0],background_ranges[1]] # get background
med = np.median(bckgrd,axis = (-1,-2)) # get median
med = med[:, np.newaxis, np.newaxis]

# crop and normalize
im_croped = im[:, image_ranges[0], image_ranges[1]] # crop images
im_norm = im_croped / med # normalise
im_norm_holo = im_norm[ref_image_num] # catch reference image
im_norm = (128*im_norm).astype(np.uint8)
save_as_multipage_tiff(im_norm, normalized_path)


if show:
    display_tiff(normalized_path, title = "Cropped and normalized original images")


##
## backpropagation
##

t= time.time()


sqrt_im_norm_holo = np.sqrt(im_norm_holo)
phase_ims, intensity_ims = parallel_backpropagation(z, sqrt_im_norm_holo, lambda_, n, pp, zero_padd, w)


print((time.time()-t))

## save results
save_as_multipage_tiff(phase_ims, phase_path)
save_as_multipage_tiff(intensity_ims, intensity_path)

if show:
    display_tiff(phase_path, title = "Intensity backpropagation")
    display_tiff(intensity_path, title = "Phase backpropagation")




