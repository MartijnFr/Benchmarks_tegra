#!/usr/bin/env python

import os

from matplotlib import pyplot as plt
import numpy as np
import kernel_tuner as kt

dir = os.path.dirname(os.path.abspath(__file__))

def tune():

    kernel_name = "grayscale"
    kernel_string = dir + "/rgb2gray.cu"

    width = np.int32(4096)
    height = np.int32(3072)
    image = 255*np.random.random((width,height,3)).astype(np.uint8)

    grayscaled = (0.229 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]).flatten().astype(np.float32);
    image_flat = image.flatten()

    args = [height, width, np.zeros_like(grayscaled), image_flat]
    answer = [None, None, grayscaled, None]

    tune_params = dict()
    tune_params["block_size_x"] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    tune_params["block_size_y"] = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    restrict = ["block_size_x * block_size_y >= 32"]

    kt.tune_kernel(kernel_name, kernel_string, (width, height), args, tune_params, answer=answer, restrictions=restrict)


if __name__ == "__main__":
    tune()
