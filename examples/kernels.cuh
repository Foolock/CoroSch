#pragma once
#include <cuda_runtime.h>

__global__ void matmul(const float* A, const float* B, float* C, int length, int num_itr);

void launch_matmul_update(const std::vector<float>& A,
                          const std::vector<float>& B,
                          std::vector<float>& C, 
                          const float* d_A,
                          const float* d_B,
                          float* d_C,
                          int length, int num_itr, cudaStream_t stream);


