__global__ void matmul(const float* A, const float* B, float* C, int length, int num_itr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = length * length;

  if (i < total_elements) {
    int row = i / length;
    int col = i % length;
    float value = 0.0f;

    for (int itr = 0; itr < num_itr; ++itr) {
      value = 0.0f;
      for (int k = 0; k < length; ++k) {
        value += A[row * length + k] * B[k * length + col];
      }
    }

    C[i] = value;
  }
}

void launch_matmul_update(const std::vector<float>& A,
                          const std::vector<float>& B,
                          std::vector<float>& C,
                          const float* d_A,
                          const float* d_B,
                          float* d_C,
                          int length, int num_itr, cudaStream_t stream) {

  size_t size = length * length * sizeof(float);

  cudaMemcpyAsync((void*)d_A, A.data(), size, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync((void*)d_B, B.data(), size, cudaMemcpyHostToDevice, stream);

  int block_size = 512;
  int grid_size = (length * length + block_size - 1) / block_size;
  matmul<<<grid_size, block_size, 0, stream>>>(d_A, d_B, d_C, length, num_itr);

  cudaMemcpyAsync(C.data(), d_C, size, cudaMemcpyDeviceToHost, stream);
}
