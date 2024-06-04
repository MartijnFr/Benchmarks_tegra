/*  Correlator.cu
 * 
 *  Copyright (C) 2012-2014  ASTRON (Netherlands Institute for Radio Astronomy)
 *  P.O. Box 2, 7990 AA Dwingeloo, The Netherlands
 * 
 *  This file is part of the LOFAR software suite.
 *  The LOFAR software suite is free software: you can redistribute it and/or
 *  modify it under the terms of the GNU General Public License as published
 *  by the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 * 
 *  The LOFAR software suite is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 * 
 *  You should have received a copy of the GNU General Public License along
 *  with the LOFAR software suite. If not, see <http://www.gnu.org/licenses/>.
 * 
 *  $Id$
 */

/*! \file
 * This file contains a CUDA implementation of the GPU kernel for the
 * correlator. It computes correlations between all pairs of stations
 * (baselines) and X,Y polarizations, including auto-correlations.
 */

// Provide the equivalent of the OpenCL swizzle feature. Obviously, the OpenCL
// \c swizzle operation is more powerful, because it's a language built-in.
#define SWIZZLE(ARG, X, Y, Z, W) make_float4((ARG).X, (ARG).Y, (ARG).Z, (ARG).W)

inline __device__ float4 operator + (float4 a, float4 b)
{
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline __device__ float4 operator - (float4 a, float4 b)
{
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline __device__ float4 operator * (float4 a, float4 b)
{
  return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

inline __device__ float4 operator / (float4 a, float4 b)
{
  return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

inline __device__ float4& operator += (float4 &a, float4 b)
{
  a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
  return a;
}

inline __device__ float4& operator -= (float4 &a, float4 b)
{
  a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
  return a;
}

inline __device__ float4& operator *= (float4 &a, float4 b)
{
  a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w;
  return a;
}

inline __device__ float4& operator /= (float4 &a, float4 b)
{
  a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w;
  return a;
}


// to distinguish complex float/double from other uses of float2/double2
typedef float2  fcomplex;
typedef double2 dcomplex;

typedef char2  char_complex;
typedef short2 short_complex;


// Operator overloads for complex values
//
// Do not implement a type with these, as it must be non-POD.
// This introduces redundant member inits in the constructor,
// causing races when declaring variables in shared memory.
inline __device__ fcomplex operator+(fcomplex a, fcomplex b)
{
  return make_float2(a.x + b.x, a.y + b.y);
}

inline __device__ dcomplex operator+(dcomplex a, dcomplex b)
{
  return make_double2(a.x + b.x, a.y + b.y);
}

inline __device__ fcomplex operator-(fcomplex a, fcomplex b)
{
  return make_float2(a.x - b.x, a.y - b.y);
}

inline __device__ dcomplex operator-(dcomplex a, dcomplex b)
{
  return make_double2(a.x - b.x, a.y - b.y);
}

inline __device__ fcomplex operator*(fcomplex a, fcomplex b)
{
  return make_float2(a.x * b.x - a.y * b.y,
                     a.x * b.y + a.y * b.x);
}

inline __device__ dcomplex operator*(dcomplex a, dcomplex b)
{
  return make_double2(a.x * b.x - a.y * b.y,
                      a.x * b.y + a.y * b.x);
}

inline __device__ fcomplex operator*(fcomplex a, float b)
{
  return make_float2(a.x * b, a.y * b);
}

inline __device__ dcomplex operator*(dcomplex a, double b)
{
  return make_double2(a.x * b, a.y * b);
}

inline __device__ fcomplex operator*(float a, fcomplex b)
{
  return make_float2(a * b.x, a * b.y);
}

inline __device__ dcomplex operator*(double a, dcomplex b)
{
  return make_double2(a * b.x, a * b.y);
}

inline __device__ dcomplex dphaseShift(double frequency, double delay)
{
  // Convert the fraction of sample duration (delayAtBegin/delayAfterEnd) to fractions of a circle.
  // Because we `undo' the delay, we need to rotate BACK.
  //
  // This needs to be done in double precision, because phi may become
  // large after we multiply a small delay and a very large freq,
  // Then we need to compute a good sin(phi) and cos(phi).
  double phi = -2.0 * delay * frequency; // -2.0 * M_PI: M_PI below in sincospi()

  dcomplex rv;
  sincospi(phi, &rv.y, &rv.x); // store (cos(), sin())
  return rv;
}

// Math definitions needed for NVRTC compilation
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Functions implemented with inline PTX to force the
// the compiler to use the desired instructions.

// multiply–accumulate: a += b * c
inline __device__ void mac(float& a, float b, float c)
{
  asm ("fma.rn.f32 %0,%1,%2,%3;" : "=f"(a) : "f"(b), "f"(c), "f"(a));
}

// Complex multiply–accumulate: a += b * c
inline __device__ void cmac(fcomplex& a, fcomplex b, fcomplex c)
{
  asm ("fma.rn.f32 %0,%1,%2,%3;" : "=f"(a.x) : "f"(b.x), "f"(c.x), "f"(a.x));
  asm ("fma.rn.f32 %0,%1,%2,%3;" : "=f"(a.y) : "f"(b.x), "f"(c.y), "f"(a.y));
  asm ("fma.rn.f32 %0,%1,%2,%3;" : "=f"(a.x) : "f"(-b.y), "f"(c.y), "f"(a.x));
  asm ("fma.rn.f32 %0,%1,%2,%3;" : "=f"(a.y) : "f"(b.y), "f"(c.x), "f"(a.y));
}

#define NR_BASELINES     (NR_STATIONS * (NR_STATIONS + 1) / 2)

#if !(NR_SAMPLES_PER_INTEGRATION % BLOCK_SIZE == 0)
#error Precondition violated: NR_SAMPLES_PER_INTEGRATION % BLOCK_SIZE == 0
#endif

typedef unsigned int uint;

typedef float2 fcomplex;
typedef float4 fcomplex2;

typedef fcomplex2 (*CorrectedDataType)[NR_STATIONS][NR_CHANNELS][NR_INTEGRATIONS][NR_SAMPLES_PER_INTEGRATION];
typedef fcomplex (*VisibilitiesType)[NR_INTEGRATIONS][NR_BASELINES][NR_CHANNELS][NR_POLARIZATIONS][NR_POLARIZATIONS];

/*!
 * Return baseline major-minor.
 *
 * Note that major >= minor >= 0 must hold.
 */
inline __device__ int baseline(int major, int minor)
{
  return major * (major + 1) / 2 + minor;
}

/*
 * Baselines are ordered like:
 *   0-0, 1-0, 1-1, 2-0, 2-1, 2-2, ...
 *
 * if 
 *   b = baseline
 *   x = stat1 (major)
 *   y = stat2 (minor)
 *   x >= y
 * then
 *   b_xy = x * (x + 1) / 2 + y
 * let
 *   u := b_x0
 * then
 *     u            = x * (x + 1) / 2
 *     8u           = 4x^2 + 4x
 *     8u + 1       = 4x^2 + 4x + 1 = (2x + 1)^2
 *     sqrt(8u + 1) = 2x + 1
 *                x = (sqrt(8u + 1) - 1) / 2
 *
 * Let us define
 *   x'(b) = (sqrt(8b + 1) - 1) / 2
 * which increases monotonically and is a continuation of y(b).
 *
 * Because y simply increases by 1 when b increases enough, we
 * can just take the floor function to obtain the discrete y(b):
 *   x(b) = floor(x'(b))
 *        = floor(sqrt(8b + 1) - 1) / 2)
 */

#undef major
inline __device__ int major(int baseline) {
  return __float2uint_rz(sqrtf(float(8 * baseline + 1)) - 0.99999f) / 2;
}

/*
 * And, of course
 *  y = b - x * (x + 1)/2
 */

#undef minor
inline __device__ int minor(int baseline, int major) {
  return baseline - ::baseline(major, 0);
}

#define SAMPLE_STATION_DIM(N) ((NR_STATIONS + (N - 1)) / N | 1)

template<int n>
inline __device__ void load_samples(int channel, int integration, int major,
                                    CorrectedDataType correctedData, void *samplesPtr)
{
  typedef float4 (*SamplesType)[n][BLOCK_SIZE][SAMPLE_STATION_DIM(n)];
  SamplesType samples = (SamplesType) samplesPtr;

  #pragma unroll 1
  for (uint i = threadIdx.x; i < BLOCK_SIZE * NR_STATIONS; i += blockDim.x)
  {
    uint time = i % BLOCK_SIZE;
    uint stat = i / BLOCK_SIZE;
    uint stat_ = 0;
    if (n == 2) {
      stat_ = stat & 1;
    } else if (n > 2) {
      stat_ = stat % n;
    }
    (*samples)[stat_][time][stat / n] = (*correctedData)[stat][channel][integration][major + time];
  }
}

template<int n>
inline __device__ void do_correlate(int x, int y, void *samplesPtr,
                                    float4* vis_r, float4* vis_i)
{
  typedef float4 (*SamplesType)[n][BLOCK_SIZE][SAMPLE_STATION_DIM(n)];
  SamplesType samples = (SamplesType) samplesPtr;

  for (uint time = 0; time < BLOCK_SIZE; time++)
  {
    for (uint i = 0; i < n*n; i++)
    {
      float4 sample_0 = (*samples)[i/n][time][x];
      float4 sample_A = (*samples)[i%n][time][y];
      float4 sample_x = SWIZZLE(sample_0,x,x,z,z);
      float4 sample_y = SWIZZLE(sample_0,y,y,w,w);
      float4 sample_z = SWIZZLE(sample_A,x,z,x,z);
      float4 sample_w = SWIZZLE(sample_A,y,w,y,w);
      vis_r[i] += sample_x * sample_z;
      vis_i[i] += sample_y * sample_z;
      vis_r[i] += sample_y * sample_w;
      vis_i[i] -= sample_x * sample_w;
    }
  }
}

template<int n>
inline __device__ void compute_do_baseline(int stat_0, int stat_A, bool *do_baseline)
{
  int stat_1 = stat_0 + 1;
  int stat_2 = stat_0 + 2;
  int stat_3 = stat_0 + 3;
  int stat_B = stat_A + 1;
  int stat_C = stat_A + 2;
  int stat_D = stat_A + 3;

  if (n == 1) {
    do_baseline[0] = stat_0 < NR_BASELINES;
  } else if (n == 2) {
    do_baseline[0] = stat_0 < NR_STATIONS;// stat_0 >= stat_A holds
    do_baseline[1] = stat_0 < NR_STATIONS && stat_0 >= stat_B;
    do_baseline[2] = stat_1 < NR_STATIONS;// stat_1 > stat_0 >= stat_A holds
    do_baseline[3] = stat_1 < NR_STATIONS;// stat_1 = stat_0 + 1 >= stat_A + 1 = stat_B holds
  } else if (n == 3) {
    do_baseline[0] = stat_0 < NR_STATIONS && stat_A < NR_STATIONS;// stat_0 >= stat_A holds
    do_baseline[1] = stat_0 < NR_STATIONS && stat_B < NR_STATIONS && stat_0 >= stat_B;
    do_baseline[2] = stat_0 < NR_STATIONS && stat_C < NR_STATIONS && stat_0 >= stat_C;
    do_baseline[3] = stat_1 < NR_STATIONS && stat_A < NR_STATIONS;// stat_1 >= stat_A holds
    do_baseline[4] = stat_1 < NR_STATIONS && stat_B < NR_STATIONS;// stat_1 >= stat_B holds
    do_baseline[5] = stat_1 < NR_STATIONS && stat_C < NR_STATIONS && stat_1 >= stat_C;
    do_baseline[6] = stat_2 < NR_STATIONS && stat_A < NR_STATIONS;// stat_2 >= stat_A holds
    do_baseline[7] = stat_2 < NR_STATIONS && stat_B < NR_STATIONS;// stat_2 >= stat_B holds
    do_baseline[8] = stat_2 < NR_STATIONS && stat_C < NR_STATIONS;// stat_2 >= stat_C holds
  } else if (n == 4) {
    do_baseline[ 0] = stat_0 < NR_STATIONS && stat_A < NR_STATIONS;// stat_0 >= stat_A holds
    do_baseline[ 1] = stat_0 < NR_STATIONS && stat_B < NR_STATIONS && stat_0 >= stat_B;
    do_baseline[ 2] = stat_0 < NR_STATIONS && stat_C < NR_STATIONS && stat_0 >= stat_C;
    do_baseline[ 3] = stat_0 < NR_STATIONS && stat_D < NR_STATIONS && stat_0 >= stat_D;
    do_baseline[ 4] = stat_1 < NR_STATIONS && stat_A < NR_STATIONS;// stat_1 >= stat_A holds
    do_baseline[ 5] = stat_1 < NR_STATIONS && stat_B < NR_STATIONS;// stat_1 >= stat_B holds
    do_baseline[ 6] = stat_1 < NR_STATIONS && stat_C < NR_STATIONS && stat_1 >= stat_C;
    do_baseline[ 7] = stat_1 < NR_STATIONS && stat_D < NR_STATIONS && stat_1 >= stat_D;
    do_baseline[ 8] = stat_2 < NR_STATIONS && stat_A < NR_STATIONS;// stat_2 >= stat_A holds
    do_baseline[ 9] = stat_2 < NR_STATIONS && stat_B < NR_STATIONS;// stat_2 >= stat_B holds
    do_baseline[10] = stat_2 < NR_STATIONS && stat_C < NR_STATIONS;// stat_2 >= stat_C holds
    do_baseline[11] = stat_2 < NR_STATIONS && stat_D < NR_STATIONS && stat_2 >= stat_D;
    do_baseline[12] = stat_3 < NR_STATIONS && stat_A < NR_STATIONS;// stat_3 >= stat_A holds
    do_baseline[13] = stat_3 < NR_STATIONS && stat_B < NR_STATIONS;// stat_3 >= stat_B holds
    do_baseline[14] = stat_3 < NR_STATIONS && stat_C < NR_STATIONS;// stat_3 >= stat_C holds
    do_baseline[15] = stat_3 < NR_STATIONS && stat_D < NR_STATIONS;// stat_3 >= stat_D holds
  }
}

inline __device__ void store_visibility(int integration,
                                        int baseline,
                                        int channel,
                                        const float4& visR, const float4& visI,
                                        VisibilitiesType visibilities)
{
  (*visibilities)[integration][baseline][channel][0][0] = make_float2(visR.x, visI.x);
  (*visibilities)[integration][baseline][channel][1][0] = make_float2(visR.y, visI.y);
  (*visibilities)[integration][baseline][channel][0][1] = make_float2(visR.z, visI.z);
  (*visibilities)[integration][baseline][channel][1][1] = make_float2(visR.w, visI.w);
}

template<int n>
__device__ void correlate_nxn(void *visibilitiesPtr, const void *correctedDataPtr)
{
  VisibilitiesType visibilities = (VisibilitiesType) visibilitiesPtr;
  CorrectedDataType correctedData = (CorrectedDataType) correctedDataPtr;

  __shared__ fcomplex2 samples[n][BLOCK_SIZE][SAMPLE_STATION_DIM(n)]; // avoid power-of-2

  int block =  blockIdx.x * blockDim.x + threadIdx.x;
  int channel = NR_CHANNELS == 1 ? 0 : blockIdx.y + 1;

  int x = ::major(block);
  int y = ::minor(block, x);

  /* NOTE: stat_0 >= stat_A holds */
  int stat_0 = n * x;
  int stat_A = n * y;

  bool compute_correlations = stat_0 < NR_STATIONS;

#if NR_INTEGRATIONS == 1
  // Fast code path for common case
  const int integration = 0; {
#else
  for (int integration = 0; integration < NR_INTEGRATIONS; integration++) {
#endif
    float4 vis_r[n*n];
    float4 vis_i[n*n];
    for (uint i = 0; i < n*n; i++) {
      vis_r[i] = make_float4(0, 0, 0, 0);
      vis_i[i] = make_float4(0, 0, 0, 0);
    }

    for (uint major = 0; major < NR_SAMPLES_PER_INTEGRATION; major += BLOCK_SIZE) {
      /* load data into local memory */
      load_samples<n>(channel, integration, major, correctedData, &samples);

      __syncthreads();

      if (compute_correlations) {
        do_correlate<n>(x, y, &samples, vis_r, vis_i);
      }

      __syncthreads();
    }

    /* write visibilities */
    /* NOTE: XY and YX polarizations have been swapped (see issue #5640) */

    bool do_baseline[n*n];
    compute_do_baseline<n>(stat_0, stat_A, do_baseline);

    for (uint i = 0; i < (n*n); i++) {
      int major = stat_0 + i/n;
      int minor = stat_A + i%n;
      int baseline = ::baseline(major, minor);
      if (baseline < NR_BASELINES && do_baseline[i]) {
        store_visibility(integration, baseline, channel, vis_r[i], vis_i[i], visibilities);
      }
    }
  }
}

extern "C" {

/*!
 * Computes correlations between all pairs of stations (baselines) and X,Y
 * polarizations. Also computes all station (and pol) auto-correlations.
 *
 * We consider the output space shaped as a triangle of S*(S-1)/2 full
 * correlations, plus S auto-correlations at the hypothenuse (S = NR_STATIONS).
 * This correlator consists of various versions, correlate_NxN, that differ in
 * used register block size. We have 1x1 (this kernel), 2x2, 3x3, and 4x4.
 * Measure, then select the fastest for your platform.
 *
 * Beyond dozens of antenna fields (exact number depends on observation,
 * software and hardware parameters), our kernels in NewCorrelator.cl are
 * significantly faster than these correlator kernels.
 *
 * \param[out] visibilitiesPtr         2D output array of visibilities. Each visibility contains the 4 polarization pairs, XX, XY, YX, YY, each of complex float type.
 * \param[in]  correctedDataPtr        3D input array of samples. Each sample contains the 2 polarizations X, Y, each of complex float type.
 *
 * Pre-processor input symbols (some are tied to the execution configuration)
 * Symbol                  | Valid Values            | Description
 * ----------------------- | ----------------------- | -----------
 * NR_STATIONS_PER_THREAD  | 1, 2, 3, 4              | the number of stations correlated by each thread
 * NR_STATIONS             | >= 1                    | number of antenna fields
 * NR_SAMPLES_PER_INTEGRATION  | multiple of BLOCK_SIZE  | number of input samples per channel
 * NR_CHANNELS             | >= 1                    | number of frequency channels per subband
 * NR_INTEGRATIONS         | >= 1                    | number of integrations to produce per block
 * Note that for > 1 channels, NR_CHANNELS-1 channels are actually processed,
 * because the second PPF has "corrupted" channel 0. (An inverse PPF can disambiguate.) \n
 * Note that if NR_CHANNELS is low (esp. 1), these kernels perform poorly.
 * Note that this kernel assumes (but does not use) NR_POLARIZATIONS == 2.
 *
 * Execution configuration:
 * - Work dim == 2  (can be 1 iff NR_CHANNELS <= 2)
 *     + Inner dim: the NxN baseline(s) the thread processes
 *     + Outer dim: the channel the thread processes
 * - Work group size: (no restrictions (but processes BLOCK_SIZE * NR_STATIONS), 1) \n
 *   Each work group loads samples from all stations to do the NxN set of correlations
 *   for one of the channels. Some threads in _NxN kernels do not write off-edge output.
 * - Global size: (>= NR_BASELINES and a multiple of work group size, number of actually processed channels)
 *
 * \note When correlating two dual-polarization station data streams for
 * stations \c ANTENNA1 and \c ANTENNA2, one computes the coherency matrix
 * \f[
 *   \bf E = \left( \begin{array}{cc}
 *                    x_1 x_2^* & x_1 y_2^* \\
 *                    y_1 x_2^* & y_1 y_2^*
 *                  \end{array}
 *           \right)
 * \f]
 * Given the signal column vector of \c ANTENNA1,
 * \f$ \bf s_1 = \left( \begin{array}{c} x_1 \\ y_1 \end{array} \right) \f$,
 * and of \c ANTENNA2,
 * \f$ \bf s_2 = \left( \begin{array}{c} x_2 \\ y_2 \end{array} \right) \f$,
 * this can also be written as 
 * \f[
 *   \bf E = \bf s_1 \cdot \bf s_2^\dagger
 * \f]
 * where \f$^\dagger\f$ indicates the hermitian conjugation and transposition.
 * That is, \f$\bf s_2^\dagger\f$ is a \e row vector with elements
 * \f$(x_2^*, y_2^*)\f$.
 * \n\n
 * In Cobalt, \c ANTENNA1 \c >= \c ANTENNA2, however, in the output Measurement
 * Set, \c ANTENNA1 \c <= \c ANTENNA2. The relation between the coherency
 * matrices is
 * \f[
 *   \bf E_{\tt ANTENNA1 \le \tt ANTENNA2}^{} = 
     \bf E_{\tt ANTENNA1 \ge \tt ANTENNA2}^\dagger
 * \f]
 * The visibilities must therefore be conjugated, \b and the \e xy and \e yx
 * polarizations must be exchanged. Hence the perhaps somewhat odd indexing in
 * the part of the code below where the output visibilities are written.
 * \n
 * Note that the convention of which stream to conjugate also depends on the
 * sign of the Fourier transform in the time-to-frequency transform. In Cobalt
 * this should be an \c FFT_FORWARD transform, which carries a minus sign in its
 * complex exponential. The conjugation used below is consistent with the
 * combination of the conjugation required by the Casa Measurement Set (de facto
 * the casa imager convention), and the \c FFTW_FORWARD time-to-frequency
 * transform.
 */

__global__ void correlate(float2 *visibilitiesPtr, const float2 *correctedDataPtr)
{
  switch (NR_STATIONS_PER_THREAD) {
    case 1:
      correlate_nxn<1>(visibilitiesPtr, correctedDataPtr);
      break;
    case 2:
      correlate_nxn<2>(visibilitiesPtr, correctedDataPtr);
      break;
    case 3:
      correlate_nxn<3>(visibilitiesPtr, correctedDataPtr);
      break;
    case 4:
      correlate_nxn<4>(visibilitiesPtr, correctedDataPtr);
      break;
  }
}

} /* extern "C" */
