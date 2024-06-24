#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define min(a,b) ((a < b) ? a : b)
#define max(a,b) ((a > b) ? a : b)

typedef struct {
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;

void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("Cuda error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void checkCudaErrors(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS error: %d\n", status);
        exit(EXIT_FAILURE);
    }
}

void matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB) {
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j) {
            double sum = 0;
            for (unsigned int k = 0; k < wA; ++k)
                sum += A[i * wA + k] * B[k * wB + j];
            C[i * wB + j] = (float)sum;
        }
}

void matrixMulGPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB) {
    // Implement matrix multiplication on GPU using CUDA and CUBLAS
    cublasHandle_t handle;
    checkCudaErrors(cublasCreate(&handle));
    float alpha = 1.0f, beta = 0.0f;
    checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, wB, hA, wA, &alpha, B, wB, A, wA, &beta, C, wB));

    checkCudaErrors(cublasDestroy(handle));
}

void randomInit(float *data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

void initializeCUDA(int &devID, int &iSizeMultiple, sMatrixSize &matrix_size) {
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaSetDevice(devID));

    printf("GPU Device %d\n", devID);
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);

    int block_size = (deviceProp.major < 2) ? 16 : 32;

    matrix_size.uiWA = 3 * block_size * iSizeMultiple;
    matrix_size.uiHA = 4 * block_size * iSizeMultiple;
    matrix_size.uiWB = 2 * block_size * iSizeMultiple;
    matrix_size.uiHB = 3 * block_size * iSizeMultiple;
    matrix_size.uiWC = 2 * block_size * iSizeMultiple;
    matrix_size.uiHC = 4 * block_size * iSizeMultiple;

    if (matrix_size.uiWA != matrix_size.uiHB || matrix_size.uiHA != matrix_size.uiHC || matrix_size.uiWB != matrix_size.uiWC) {
        printf("ERROR: Matrix sizes do not match!\n");
        exit(EXIT_FAILURE);
    }
}

int matrixMultiply(int devID, sMatrixSize &matrix_size) {
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
    
    srand(2006);

    // Allocate host memory for matrices
    unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);
    unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);

    // Initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);
    
    // Allocate device memory for matrices
    unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float *h_CUBLAS = (float *)malloc(mem_size_C);
    float *reference = (float *)malloc(mem_size_C);
    
    // Copy host memory to device memory
    float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc((void **)&d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void **)&d_B, mem_size_B));
    checkCudaErrors(cudaMalloc((void **)&d_C, mem_size_C));
    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));

    // Multiply matrices on GPU and CPU
    matrixMulGPU(d_C, d_A, d_B, matrix_size.uiHA, matrix_size.uiWA, matrix_size.uiWB);
    matrixMulCPU(reference, h_A, h_B, matrix_size.uiHA, matrix_size.uiWA, matrix_size.uiWB);
    checkCudaErrors(cudaMemcpy(h_CUBLAS, d_C, mem_size_C, cudaMemcpyDeviceToHost));

    // Compare CPU and GPU results
    float loss = 0.f;
    bool match = true;
    float max_loss = 0.f;
    float diff_a, diff_b;
    for (unsigned int i = 0; i < size_C; ++i) {
        auto l = fabs(h_CUBLAS[i] - reference[i]);
        match = match && (l < 1e-3);
        loss += l;
        if (max_loss < l) {
            max_loss = l;
            diff_a = h_CUBLAS[i];
            diff_b = reference[i];
        }
    }

    if (match)
        printf("Results match!\n");
    else
        printf("Results do not match!\n");

    printf("Loss: %f, Max Loss: %f, Diff A: %f, Diff B: %f\n", loss / size_C, max_loss, diff_a, diff_b);

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_CUBLAS);
    free(reference);
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    cudaDeviceReset();

    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    printf("[Matrix Multiply CUBLAS] - Starting...\n");

    int devID = 0, sizeMult = 5;
    sMatrixSize matrix_size;

    initializeCUDA(devID, sizeMult, matrix_size);
    int matrix_result = matrixMultiply(devID, matrix_size);

    return matrix_result;
}
