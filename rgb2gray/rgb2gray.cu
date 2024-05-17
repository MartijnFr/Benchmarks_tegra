


__global__ void grayscale(int h, int w, float* output, uchar3* input) {
	int i = threadIdx.y + blockIdx.y * block_size_y;
	int j = threadIdx.x + blockIdx.x * block_size_x;

	if (j < w && i < h) {

		uchar3 c = input[i*w+j];

		output[i*w+j] = 0.299f*c.x + 0.587f*c.y + 0.114f*c.z;

	}
}
