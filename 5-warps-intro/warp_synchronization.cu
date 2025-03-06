#include <stdio.h>
#include <cuda_runtime.h>

__global__ void warpSyncKernel(float *A, float *B, float *C, int N) {
    __shared__ float sharedA[32];
    __shared__ float sharedB[32];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpIdx = threadIdx.x % 32;

    if(idx < N) {
        sharedA[warpIdx] = A[idx];
        sharedB[warpIdx] = B[idx];
        __syncthreads(); // Synchronize within the block

        C[idx] = sharedA[warpIdx] + sharedB[warpIdx];
    }
}

int main() {
    int N = 64; // Reduced size for easier understanding
    int threadsPerBlock = 32; // One warp per block for clarity
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));
    float *h_C = (float*)malloc(N * sizeof(float));

    for(int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f; // Using index as value for better visualization
        h_B[i] = i * 0.5f; // Using half of index as value
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    warpSyncKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results in a more readable format
    printf("Results (showing A[i] + B[i] = C[i]):\n");
    for (int i = 0; i < N; i++) {
        printf("A[%d]=%.1f + B[%d]=%.1f = C[%d]=%.1f\n", 
               i, h_A[i], i, h_B[i], i, h_C[i]);
    }

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
