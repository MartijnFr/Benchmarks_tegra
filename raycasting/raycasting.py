#!/usr/bin/env python
""" Tuning script for raycasting kernel

    This script tunes cuda_ray_marching kernel from RangeLibc (https://github.com/kctess5/range_libc)

    This library is used in for localization in autonomous vehicles and robotics, see for example: https://github.com/mit-racecar/particle_filter/

    Walsh, C. H., & Karaman, S. (2018, May).
    CDDT: Fast approximate 2d ray casting for accelerated localization.
    In 2018 IEEE International Conference on Robotics and Automation (ICRA) (pp. 3677-3684). IEEE.

"""
import os

from matplotlib import pyplot as plt
import numpy as np
import kernel_tuner as kt


dir = os.path.dirname(os.path.abspath(__file__))

def tune():

    kernel_name = "cuda_ray_marching"
    kernel_string = dir + "/raycasting.cu"

    # setup kernel input and output data

    # read input image
    image = np.array(plt.imread(dir + '/gigantic_map.png')).astype(np.float32)

    # convert rgb2gray
    distMap = (0.229 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]).flatten();
    width = np.int32(image.shape[1])
    height = np.int32(image.shape[0])

    num_vals = 1e5
    num_casts = np.int32(num_vals)
    vals = np.random.random((3,num_casts)).astype(np.float32)
    vals[0,:] *= (width - 2.0)
    vals[1,:] *= (height - 2.0)
    vals[0,:] += 1.0
    vals[1,:] += 1.0
    vals[2,:] *= np.pi * 2.0
    ins = vals
    outs = np.zeros(num_casts, dtype=np.float32)

    max_range = np.float32(500)

    args = [ins, outs, distMap, width, height, max_range, num_casts]

    tune_params = dict()
    tune_params["block_size_x"] = [32, 64, 128, 256, 512, 1024]

    # call the tuner
    kt.tune_kernel(kernel_name, kernel_string, num_casts, args, tune_params)



if __name__ == "__main__":
    tune()
