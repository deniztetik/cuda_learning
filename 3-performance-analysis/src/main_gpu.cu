#include <stdio.h>
#include <cuda_runtime.h>
#include <ctime>

__global__ void vectorAddGPU(float *A, float *B, float *C, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    clock_t total_program_start = clock();

    size_t N = 200000000; // 200 million elements (800MB per array)
    size_t size = N * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize vectors
    for(int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaError_t err;

    err = cudaMalloc(&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for A: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc(&d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for B: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc(&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for C: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        exit(EXIT_FAILURE);
    }

    // Start timing memory transfer (H2D)
    clock_t mem_h2d_start = clock();
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    clock_t mem_h2d_end = clock();
    double mem_h2d_time = ((double)(mem_h2d_end - mem_h2d_start))/CLOCKS_PER_SEC;

    // Start timing kernel execution
    clock_t kernel_start = clock();
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C);
        exit(EXIT_FAILURE);
    }
    clock_t kernel_end = clock();
    double kernel_time = ((double)(kernel_end - kernel_start))/CLOCKS_PER_SEC;

    // Start timing memory transfer (D2H)
    clock_t mem_d2h_start = clock();
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    clock_t mem_d2h_end = clock();
    double mem_d2h_time = ((double)(mem_d2h_end - mem_d2h_start))/CLOCKS_PER_SEC;

    clock_t total_program_end = clock();
    double total_program_time = ((double)(total_program_end - total_program_start))/CLOCKS_PER_SEC;

    printf("GPU Timings:\n");
    printf("  Host to Device Memory Transfer: %lf seconds\n", mem_h2d_time);
    printf("  Kernel Execution: %lf seconds\n", kernel_time);
    printf("  Device to Host Memory Transfer: %lf seconds\n", mem_d2h_time);
    printf("  GPU Operations Total: %lf seconds\n", mem_h2d_time + kernel_time + mem_d2h_time);
    printf("  Total Program Time (including all operations): %lf seconds\n", total_program_time);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
