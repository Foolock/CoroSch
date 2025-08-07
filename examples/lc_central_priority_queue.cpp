#include "coro_scheduler.hpp"
#include "kernels.cuh"

cs::Task gpu_mm_cpu_reduce_coro(cs::SchedulerCentralPriorityQueue& sch, const int num_itr, const int length) {
  size_t size = length * length * sizeof(float);
  int block_size = 512;
  int grid_size = (length * length + block_size - 1) / block_size;

  float* A = new float[size];
  float* B = new float[size];
  float* C = new float[size];

  std::fill(A, A + size, 1.0f);
  std::fill(B, B + size, 1.0f);
  std::fill(C, C + size, 0.0f);

  float *d_A, *d_B, *d_C;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

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
                         block_size,
                         grid_size,
                         size,
                         length, num_itr, stream);

    // Async wait
    while (cudaStreamQuery(stream) != cudaSuccess) {
      co_await sch.suspend();
    }

    float partial_sum = std::accumulate(C, C + size, 0.0f);
    cpu_sum += partial_sum;
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaStreamDestroy(stream);

  delete[] A;
  delete[] B;
  delete[] C;

  float expected_total = static_cast<float>(num_itr * length * length * length);
  if (std::abs(cpu_sum - expected_total) > 1e-3f) {
    std::cerr << "❌ Reduction result mismatch: expected " << expected_total
              << ", got " << cpu_sum << "\n";
    co_return false;
  }

  co_return true;
}

int main(int argc, char *argv[]) {

  if(argc != 5) {
    std::cerr << "usage: ./example/linear_chain_tf num_itr length num_nodes num_threads\n";
    std::exit(EXIT_FAILURE);
  }

  int num_itr = std::atoi(argv[1]);
  int length = std::atoi(argv[2]);
  int num_nodes = std::atoi(argv[3]);
  int num_threads = std::atoi(argv[4]);

  std::cout << "--------------------\n";
  std::cout << "num_itr = " << num_itr << ", length = " << length << ", num_nodes = " << num_nodes << ", num_threads = " << num_threads << "\n";

  cs::SchedulerCentralPriorityQueue coro_scheduler(num_threads);

  std::vector<cs::Task*> tasks(num_nodes);

  // emplace tasks
  for(int i = 0; i < num_nodes; i++) {
    tasks[i] = coro_scheduler.emplace(
      gpu_mm_cpu_reduce_coro(coro_scheduler, num_itr, length).get_handle()
    );
  }

  // build dependencies
  for(int i = 0; i < num_nodes - 1; i++) {
    tasks[i]->precede(tasks[i + 1]);
  }

  auto start = std::chrono::steady_clock::now();
  coro_scheduler.schedule();
  coro_scheduler.wait();
  auto end = std::chrono::steady_clock::now();
  size_t coro_runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  for(auto& task : tasks) {
    if (!task->is_done()) {
      std::cerr << "❌ calculation is wrong!\n";
      std::exit(EXIT_FAILURE);
    }
  }

  std::cout << "✅ all calculations correct\n";
  std::cout << "coro runtime = " << coro_runtime << "ms\n\n";

  return 0;
}












