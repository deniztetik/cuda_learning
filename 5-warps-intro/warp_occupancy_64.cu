#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that performs a computation
__global__ void computeKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simple computation to keep threads busy
        float value = input[idx];
        for (int i = 0; i < 100; i++) {
            value = sinf(value) * cosf(value);
        }
        output[idx] = value;
    }
}

int main() {
    // Problem size
    int n = 1 << 20; // 1M elements
    size_t bytes = n * sizeof(float);

    // Allocate host memory
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);

    // Initialize input data
    for (int i = 0; i < n; i++) {
        h_input[i] = (float)i;
    }

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    // Copy data to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Block size of 64 threads (2 warps per block)
    int blockSize = 64;
    int gridSize = (n + blockSize - 1) / blockSize;

    printf("Executing kernel with %d threads per block (%d warps per block)\n",
           blockSize, blockSize / 32);
    printf("Grid size: %d blocks\n", gridSize);
    printf("Total threads: %d\n", gridSize * blockSize);

    // Launch kernel and measure time
    cudaEventRecord(start);
    computeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    cudaEventRecord(stop);

    // Wait for kernel to finish
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // Copy results back to host
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    // Print performance metrics
    printf("\nPerformance Metrics:\n");
    printf("Execution time: %.3f ms\n", milliseconds);
    printf("Throughput: %.2f GB/s\n", (2.0 * bytes) / (milliseconds * 1.0e6));

    // Print occupancy information
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    printf("\nDevice Information:\n");
    printf("Device name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);

    printf("\nOccupancy Analysis (Block Size 64):\n");
    printf("Warps per block: %d\n", blockSize / 32);
    printf("Theoretical max blocks per SM: %d\n", prop.maxThreadsPerMultiProcessor / blockSize);
    printf("Theoretical max warps per SM: %d\n", prop.maxThreadsPerMultiProcessor / 32);
    printf("Theoretical occupancy: %.2f%%\n",
           (float)(blockSize / 32) * (prop.maxThreadsPerMultiProcessor / blockSize) /
           (prop.maxThreadsPerMultiProcessor / 32) * 100.0f);

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_input);
    free(h_output);

    return 0;
}