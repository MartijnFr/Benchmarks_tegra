#define fcomplex float2

typedef fcomplex OutputDataType[NR_CHANNELS][NR_SAMPLES_PER_CHANNEL][NR_STATIONS * NR_POLARIZATIONS];

typedef fcomplex InputDataType[NR_SAMPLES_PER_CHANNEL][NR_STATIONS * NR_POLARIZATIONS][NR_CHANNELS];

//typedef __constant const float BandPassWeightsType[NR_CHANNELS];
__constant__ float bandPassWeights[NR_CHANNELS];

extern "C"
__global__ void applyBandPass(OutputDataType outputData, const InputDataType inputData) {
  uint minor = threadIdx.x;
  uint major = threadIdx.y;
  uint station_pol = block_size_x * blockIdx.x;
  uint channel = block_size_y * blockIdx.y;

  float weight = bandPassWeights[channel + major];

  for (uint time = 0; time < NR_SAMPLES_PER_CHANNEL; time ++) {
    fcomplex sample;

    if (NR_STATIONS * NR_POLARIZATIONS % block_size_x == 0 || station_pol + minor < NR_STATIONS * NR_POLARIZATIONS)
      sample = inputData[time][station_pol + minor][channel + major];

    sample.x *= weight;
    sample.y *= weight;

    if (NR_STATIONS * NR_POLARIZATIONS % block_size_x == 0 || station_pol + minor < NR_STATIONS * NR_POLARIZATIONS)
      outputData[channel + major][time][station_pol + minor] = sample;
  }
}
