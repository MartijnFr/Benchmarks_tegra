#!/usr/bin/env python
""" Tuning script for raycasting kernel

    This script tunes cuda_ray_marching kernel from RangeLibc (https://github.com/kctess5/range_libc)

    This library is used in for localization in autonomous vehicles and robotics, see for example: https://github.com/mit-racecar/particle_filter/

    Walsh, C. H., & Karaman, S. (2018, May).
    CDDT: Fast approximate 2d ray casting for accelerated localization.
    In 2018 IEEE International Conference on Robotics and Automation (ICRA) (pp. 3677-3684). IEEE.

"""
import os
import sys

from matplotlib import pyplot as plt
import numpy as np
import kernel_tuner as kt

from kernel_tuner.observers.ncu import NCUObserver

dir = os.path.dirname(os.path.abspath(__file__))

def tune(use_profiler=False):

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

    if not os.path.isfile("input_values.npy"):
        vals = np.random.random((3,num_casts)).astype(np.float32)
        vals[0,:] *= (width - 2.0)
        vals[1,:] *= (height - 2.0)
        vals[0,:] += 1.0
        vals[1,:] += 1.0
        vals[2,:] *= np.pi * 2.0
        np.save("input_values", vals, allow_pickle=False)

    vals = np.load("input_values.npy")

    ins = vals

    outs = np.zeros(num_casts, dtype=np.float32)

    max_range = np.float32(500)

    args = [ins, outs, distMap, width, height, max_range, num_casts]

    tune_params = dict()
    tune_params["block_size_x"] = [32, 64, 128, 256, 512, 1024]

    observers = []
    metrics = {}

    if use_profiler:
        ncu_metrics = ["dram__bytes.sum",                                       # Counter         byte            # of bytes accessed in DRAM
                       "dram__bytes_read.sum",                                  # Counter         byte            # of bytes read from DRAM
                       "dram__bytes_write.sum",                                 # Counter         byte            # of bytes written to DRAM
                       "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",   # Counter         inst            # of FADD thread instructions executed where all predicates were true
                       "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",   # Counter         inst            # of FFMA thread instructions executed where all predicates were true
                       "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",   # Counter         inst            # of FMUL thread instructions executed where all predicates were true
                       "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum",   # Counter         inst            # of single-precision floating-point thread instructions executed
                                                                                                                  # where all predicates were true
                      ]

        ncuobserver = NCUObserver(metrics=ncu_metrics)
        observers=[ncuobserver]

        def total_fp32_flops(p):
            return p["smsp__sass_thread_inst_executed_op_fp32_pred_on.sum"]

        metrics = dict()
        metrics["GFLOP/s"] = lambda p: (total_fp32_flops(p) / 1e9) / (p["time"]/1e3)
        metrics["total GFLOP/s"] = lambda p: total_fp32_flops(p)

    # call the tuner
    kt.tune_kernel(kernel_name, kernel_string, num_casts, args, tune_params, metrics=metrics, observers=observers)



if __name__ == "__main__":
    if len(sys.argv) > 0 and "--use_profiler" in sys.argv:
        tune(use_profiler=True)
    else:
        tune()
