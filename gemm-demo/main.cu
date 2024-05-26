#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "main.cuh"

#define BLOCK_SIZE 16

int main() {
    int n = 4096;

    float *cpu_a, *cpu_b, *cpu_c;

    cpu_a = (float *) malloc(n * n * sizeof(float));
    cpu_b = (float *) malloc(n * n * sizeof(float));
    cpu_c = (float *) malloc(n * n * sizeof(float));

    float *gpu_a, *gpu_b, *gpu_c;

    printf("Matrix multiplication on %d x %d matrices\n", n, n);

    cudaMalloc((void **) &gpu_a, n * n * sizeof(float));

    cudaMalloc((void **) &gpu_b, n * n * sizeof(float));

    cudaMalloc((void **) &gpu_c, n * n * sizeof(float));

    // initialize matrices
    for (int i = 0; i < n * n; i++) {
        cpu_a[i] = 1;
        cpu_b[i] = 2;
    }

    cudaMemcpy(gpu_a, cpu_a, n * n * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(gpu_b, cpu_b, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // launch the kernel
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matrix_multiply<<<grid_dim, block_dim>>>(gpu_a, gpu_b, gpu_c, n);

    // copy the result back to the host
    cudaMemcpy(cpu_c, gpu_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // check the result
    for (int i = 0; i < n * n; i++) {
        if ((int)cpu_c[i] != n * 2) {
            printf("Error: c[%d] = %f\n", i, cpu_c[i]);
            return 1;
        }
    }

    printf("Success!\n");

    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);

    return 0;
}