#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
void gemm_lowbit_cuda(at::Tensor a, at::Tensor b, at::Tensor c, int M, int N, int K);
