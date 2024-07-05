#!/usr/bin/env python
import os
import kernel_tuner
import numpy as np

# return the result of a/b rounded to the nearest integer value
def ceilDiv(a, b):
    return int((a + b - 1) / b)

# returns the nearest multiple of b that is greater than or equal to a
def roundUp(a, b):
    return int((a + b - 1) / b) * b

current_dir = os.path.dirname(os.path.realpath(__file__))
cp = [f"-I{current_dir}"]

def tune():

    #the kernel to tune
    kernel_file = "bandpass.cu"
    with open(f"{os.path.dirname(os.path.realpath(__file__))}/{kernel_file}", 'r') as f:
        kernel_string = f.read()

    #kernel parameters
    nr_stations = 55
    nr_channels = 480
    nr_samples_per_channel = 512
    nr_samples_per_subband = nr_samples_per_channel
    nr_polarizations = 2

    #setup kernel compile-time constants
    compiler_options = [f"-DNR_STATIONS={nr_stations}",
                        f"-DNR_CHANNELS={nr_channels}",
                        f"-DNR_SAMPLES_PER_CHANNEL={nr_samples_per_channel}",
                        f"-DNR_SAMPLES_PER_SUBBAND={nr_samples_per_subband}",
                        f"-DNR_POLARIZATIONS={nr_polarizations}"]

    kernel_name = "applyBandPass"

    #setup test data
    output_data = np.zeros(nr_channels*nr_samples_per_channel*nr_stations*nr_polarizations, dtype=np.complex64)
    input_data = np.random.random(nr_samples_per_channel*nr_stations*nr_polarizations*nr_channels*2).astype(np.float32)
    bandpass_weights = np.random.random(nr_channels).astype(np.float32)
    kernel_arguments = [output_data, input_data]

    #setup tunable parameters
    min_threads_per_block = 32
    max_threads_per_block = 1024

    tune_params = {}
    tune_params["block_size_x"] = [1, 2, 4, 8, 16, 32]
    tune_params["block_size_y"] = [1, 2, 4, 8, 16, 32]

    restrict = [f"block_size_x*block_size_y>={min_threads_per_block}", f"block_size_x*block_size_y<={max_threads_per_block}"]

    problem_size = (nr_stations*nr_polarizations, nr_channels)

    metrics = {}
    metrics["GFLOP/s"] = lambda p: ((2*nr_samples_per_channel*nr_stations*nr_polarizations*nr_channels) / 1e9) / (p["time"] / 1e3)

    results, env = kernel_tuner.tune_kernel(kernel_name, kernel_string, problem_size,
                                            kernel_arguments, tune_params, restrictions=restrict,
                                            metrics=metrics, cmem_args=dict(bandPassWeights=bandpass_weights), lang="CUDA", compiler_options=compiler_options+cp)


if __name__ == "__main__":
    tune()
