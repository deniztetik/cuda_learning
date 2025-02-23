// Include standard input/output library for printf
#include <stdio.h>

// __global__ indicates this is a CUDA kernel function that runs on the GPU
// This function can be called from CPU (host) code and runs on the GPU (device)
__global__ void helloCUDA(float f)
{
    // threadIdx.x gives us the current thread's ID in the x dimension
    // Each thread has its own unique ID
    printf("Hello thread %d, f=%f\n", threadIdx.x, f);
}

int main()
{
    // Launch the CUDA kernel with <<<blocks, threads>>> syntax
    // <<<1,1>>> means: launch 1 block with 1 thread
    // 1.2345f is passed as parameter 'f' to the kernel
    helloCUDA<<<1, 1>>>(1.2345f);

    // Wait for GPU to finish before accessing results
    // Without this, the program might end before the printf executes
    cudaDeviceSynchronize();

    return 0;
}