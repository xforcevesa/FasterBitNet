#include "main.cuh"

__global__ void matrix_multiply(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;


    if (i < n && j < n) {
        float sum = 0;
        for (int k = 0; k < n; k++) {
            sum += a[i * n + k] * b[k * n + j];

        }
        c[i * n + j] = sum;
    }
}
