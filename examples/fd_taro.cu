#include "taro.hpp"
#include "taro/await/cuda.hpp"
#include "kernels.cuh"

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

  taro::Taro taro{static_cast<size_t>(num_threads)}; // number of threads
  auto cuda = taro.cuda_await(8); // number of cuda streams

  std::vector<taro::TaskHandle> tasks;
  std::vector<bool> done(num_nodes, false);

  // emplace tasks
  for(int i = 0; i < num_nodes; i++) {
    tasks.emplace_back(taro.emplace([length, num_itr, i, &cuda, &done]() -> taro::Coro {

      int size = length * length * sizeof(float);
      int block_size = 512;
      int grid_size = (length * length + block_size - 1) / block_size;

      float* A = new float[size];
      float* B = new float[size];
      float* C = new float[size];

      std::fill(A, A + size, 1.0f);
      std::fill(B, B + size, 1.0f);
      std::fill(C, C + size, 0.0f);

      float *d_A, *d_B, *d_C;

      cudaMalloc(&d_A, size);
      cudaMalloc(&d_B, size);
      cudaMalloc(&d_C, size);

      float cpu_sum = 0.0f;

      for (int itr = 0; itr < num_itr; ++itr) {

        // cuda.until_polling
        co_await cuda.until_polling([=](cudaStream_t stream) { // polling method   

          launch_matmul_update(A,
                               B,
                               C,
                               d_A,
                               d_B,
                               d_C,
                               block_size,
                               grid_size,
                               size,
                               length, num_itr, stream);
        });

        // Reduction on CPU
        float partial_sum = std::accumulate(C, C + size, 0.0f);

        cpu_sum += partial_sum;
      }

      cudaFree(d_A);
      cudaFree(d_B);
      cudaFree(d_C);

      delete[] A;
      delete[] B;
      delete[] C;

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

  auto start = std::chrono::steady_clock::now();
  taro.schedule();
  taro.wait();
  auto end = std::chrono::steady_clock::now();
  size_t taro_runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  // check result
  for (auto d : done) {
    if (!d) {
      std::cerr << "❌ calculation is wrong!\n";
      std::exit(EXIT_FAILURE);
    }
  }

  std::cout << "✅ all calculations correct\n";
  std::cout << "taro runtime = " << taro_runtime << "ms\n\n";

  return 0;
}





















