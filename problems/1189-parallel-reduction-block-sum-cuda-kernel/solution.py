#include <cuda_runtime.h>

__global__ void reduceKernel(const float* in, float* out, int N) {
    extern __shared__ float sdata[];          // the classroom table
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (i < N) ? in[i] : 0.0f;      // grab a coin, or a zero token
    __syncthreads();                          // wait for everyone

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];   // pair up and add
        __syncthreads();                      // wait before the next round
    }

    if (tid == 0) atomicAdd(out, sdata[0]);   // turnstile to the grand total
}

void solve(const float* input, float* output, int N) {
    if (N <= 0) { *output = 0.0f; return; }

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));

    cudaMemcpy(d_in, input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(float));

    int threads = 256;                        // must be a power of two
    int blocks  = (N + threads - 1) / threads;
    reduceKernel<<<blocks, threads, threads * sizeof(float)>>>(d_in, d_out, N);

    cudaMemcpy(output, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}