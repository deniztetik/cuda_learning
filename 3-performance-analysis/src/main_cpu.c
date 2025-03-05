#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void vectorAddCPU(float *A, float *B, float *C, size_t N) {
    for(size_t i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    clock_t total_program_start = clock();

    size_t N = 200000000; // 200 million elements
    size_t size = N * sizeof(float);

    // Allocate memory
    float *A = malloc(size);
    float *B = malloc(size);
    float *C = malloc(size);

    // Initialize vectors
    for(size_t i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // Start timing the computation
    clock_t compute_start = clock();
    vectorAddCPU(A, B, C, N);
    clock_t compute_end = clock();
    double compute_time = ((double)(compute_end - compute_start))/CLOCKS_PER_SEC;

    clock_t total_program_end = clock();
    double total_program_time = ((double)(total_program_end - total_program_start))/CLOCKS_PER_SEC;

    printf("CPU Timings:\n");
    printf("  Computation Time: %lf seconds\n", compute_time);
    printf("  Total Program Time (including all operations): %lf seconds\n", total_program_time);

    // Optional: Verify results
    // for(size_t i = 0; i < N; i++) {
    //     if(C[i] != 3.0f) {
    //         printf("Verification failed at index %zu!\n", i);
    //         break;
    //     }
    // }

    free(A);
    free(B);
    free(C);
    return 0;
}