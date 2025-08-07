#include "taro.hpp"
#include "taro/await/cuda.hpp"
#include "kernels.cuh"

template <typename T>
void print_type(T&&) {
  std::cout << __PRETTY_FUNCTION__ << '\n';  // or __FUNCSIG__ on MSVC
}

bool gpu_mm_cpu_reduce(const int num_itr, const int length, cudaStream_t stream, taro::cudaAwait& cuda) {
  size_t size = length * length * sizeof(float);

  std::vector<float> A(length * length, 1.0f);
  std::vector<float> B(length * length, 1.0f);
  std::vector<float> C(length * length, 0.0f);

  float *d_A, *d_B, *d_C;

  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  float cpu_sum = 0.0f;

  for (int itr = 0; itr < num_itr; ++itr) {

    launch_matmul_update(A,
                         B,
                         C,
                         d_A,
                         d_B,
                         d_C,
                         length, num_itr, stream);

    // cudaStreamSync;
    cudaStreamSynchronize(stream);

    // Reduction on CPU
    float partial_sum = std::accumulate(C.begin(), C.end(), 0.0f);
    cpu_sum += partial_sum;
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Validation
  float expected_total = static_cast<float>(num_itr * length * length * length);
  if (std::abs(cpu_sum - expected_total) > 1e-3f) {
    std::cerr << "❌ Reduction result mismatch: expected " << expected_total
              << ", got " << cpu_sum << "\n";
    return false;
  }

  return true;
}


int main(int argc, char *argv[]) {

  if(argc != 5) {
    std::cerr << "usage: ./example/lc_taro num_itr length num_nodes num_threads\n";
    std::exit(EXIT_FAILURE);
  }

  int num_itr = std::atoi(argv[1]);
  int length = std::atoi(argv[2]);
  int num_nodes = std::atoi(argv[3]);
  int num_threads = std::atoi(argv[4]);

  std::cout << "--------------------\n";
  std::cout << "num_itr = " << num_itr << ", length = " << length << ", num_nodes = " << num_nodes << ", num_threads = " << num_threads << "\n";

  taro::Taro taro{4}; // number of threads
  auto cuda = taro.cuda_await(4); // number of cuda streams

  std::vector<taro::TaskHandle> tasks;
  std::vector<bool> done(num_nodes, false);

  // emplace tasks
  for(int i = 0; i < num_nodes; i++) {
    tasks.emplace_back(taro.emplace([length, num_itr, i, &cuda, &done]() -> taro::Coro {

      size_t size = length * length * sizeof(float);
      int block_size = 512;
      int grid_size = (length * length + block_size - 1) / block_size;

      std::vector<float> A(length * length, 1.0f);
      std::vector<float> B(length * length, 1.0f);
      std::vector<float> C(length * length, 0.0f);

      float *d_A, *d_B, *d_C;

      cudaMalloc(&d_A, size);
      cudaMalloc(&d_B, size);
      cudaMalloc(&d_C, size);

      float cpu_sum = 0.0f;

      for (int itr = 0; itr < num_itr; ++itr) {

        cudaMemcpy((void*)d_A, A.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_B, B.data(), size, cudaMemcpyHostToDevice);

        matmul<<<grid_size, block_size, 0>>>(d_A, d_B, d_C, length, num_itr);

        cudaMemcpy(C.data(), d_C, size, cudaMemcpyDeviceToHost);

        // Reduction on CPU
        float partial_sum = std::accumulate(C.begin(), C.end(), 0.0f);
        cpu_sum += partial_sum;
      }

      cudaFree(d_A);
      cudaFree(d_B);
      cudaFree(d_C);

      // Validation
      float expected_total = static_cast<float>(num_itr * length * length * length);
      if (std::abs(cpu_sum - expected_total) > 1e-3f) {
        std::cerr << "❌ Reduction result mismatch: expected " << expected_total
                  << ", got " << cpu_sum << "\n";
        done[i] = false;
      }

      done[i] = true;
    }));
  }

  // build dependencies
  for(int i = 0; i < num_nodes - 1; i++) {
    tasks[i].precede(tasks[i + 1]);
  }

  taro.schedule();
  taro.wait();

  // check result
  for (auto d : done) {
    if (!d) {
      std::cerr << "❌ calculation is wrong!\n";
      std::exit(EXIT_FAILURE);
    }
  }

  std::cout << "✅ all calculations correct\n";

  return 0;
}





















