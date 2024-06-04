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

def tune():

    #the kernel to tune
    kernel_file = "lofar-correlator.cu"
    with open(f"{os.path.dirname(os.path.realpath(__file__))}/{kernel_file}", 'r') as f:
        kernel_string = f.read()

    #kernel parameters
    nr_stations = 55
    nr_channels = 480
    nr_integrations = 1
    nr_samples_per_integration = 3072
    nr_polarizations = 2
    nr_baselines = int((nr_stations * (nr_stations + 1) / 2))
    max_threads_per_block = 1024
    preferred_multiple = 64
    block_size = 16

    #setup kernel compile-time constants
    compiler_options = [f"-DNR_STATIONS={nr_stations}",
                        f"-DNR_CHANNELS={nr_channels}",
                        f"-DNR_INTEGRATIONS={nr_integrations}",
                        f"-DNR_SAMPLES_PER_INTEGRATION={nr_samples_per_integration}",
                        f"-DNR_POLARIZATIONS={nr_polarizations}",
                        f"-DBLOCK_SIZE={block_size}"]

    #setup tunable parameters
    tune_params = {}
    tune_params["block_size_x"] = [preferred_multiple*i for i in range(1,20)]
    tune_params["NR_STATIONS_PER_THREAD"] = [1, 2, 3, 4]

    #setup valid configurations
    def compute_nr_threads(nr_stations_per_thread):
        nr_macro_stations = ceilDiv(nr_stations, nr_stations_per_thread)
        nr_blocks = int(nr_macro_stations * (nr_macro_stations + 1) / 2)
        nr_passes = ceilDiv(nr_blocks, max_threads_per_block)
        nr_threads = ceilDiv(nr_blocks, nr_passes)
        nr_threads = roundUp(nr_threads, preferred_multiple)
        return nr_threads

    def config_valid(p):
        return p["block_size_x"] == compute_nr_threads(p["NR_STATIONS_PER_THREAD"])

    restrict = config_valid

    #kernel arguments
    input_size = (nr_stations, nr_channels, nr_integrations, nr_samples_per_integration)
    output_size = (nr_integrations, nr_baselines, nr_channels, nr_polarizations, nr_polarizations)
    input_data = np.zeros(np.prod(input_size) * 2).astype(np.complex64)
    output_data = np.zeros(np.prod(output_size)).astype(np.complex64)
    arguments = [output_data, input_data]

    #setup metrics
    metrics = {}
    total_flops = nr_integrations * 8 * nr_stations * nr_stations / 2 * nr_polarizations * nr_polarizations * nr_channels * nr_samples_per_integration
    total_flops = total_flops / 1e9 #gflops
    metrics["GFLOP/s"] = lambda p: total_flops / (p["time"] / 1000.0)
    nr_visibilities = nr_samples_per_integration * nr_baselines * nr_channels
    nr_visibilities = nr_visibilities / 1e9 #gvis
    metrics["GVIS/s"] = lambda p: (nr_visibilities) / (p["time"] / 1000.0)

    #start tuning
    def compute_problem_size(nr_stations_per_thread):
        nr_macro_stations = ceilDiv(nr_stations, nr_stations_per_thread)
        nr_blocks = int(nr_macro_stations * (nr_macro_stations + 1) / 2)
        nr_passes = ceilDiv(nr_blocks, max_threads_per_block)
        nr_usable_channels = max(nr_channels - 1, 1)
        return (nr_passes, nr_usable_channels)

    results, env = kernel_tuner.tune_kernel("correlate", kernel_string,
                    problem_size=lambda p: compute_problem_size(p["NR_STATIONS_PER_THREAD"]),
                    arguments=arguments, tune_params=tune_params,
                    restrictions=restrict,
                    verbose=True, metrics=metrics, iterations=32,
                    grid_div_x=[], grid_div_y=[],
                    compiler_options=compiler_options,
                    objective="GVIS/s", objective_higher_is_better=True)

    return results, env


if __name__ == "__main__":
    results, env = tune()
