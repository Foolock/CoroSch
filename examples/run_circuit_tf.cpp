#include "taskflow/taskflow.hpp"
#include "graph.hpp"
#include "kernels.cuh"

bool gpu_mm_cpu_reduce(const int num_itr, const int length) {
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

    // cudaStreamSync;
    cudaStreamSynchronize(stream);

    // Reduction on CPU
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
    std::cerr << "usage: ./example/run_circuit_coro num_itr length circuit_file num_threads\n";
    std::exit(EXIT_FAILURE);
  }

  int num_itr = std::atoi(argv[1]);
  int length = std::atoi(argv[2]);
  std::string circuit_file = argv[3];
  int num_threads = std::atoi(argv[4]);

  std::cout << "--------------------\n";
  std::cout << "num_itr = " << num_itr << ", length = " << length << ", circuit_file = " << circuit_file << ", num_threads = " << num_threads << "\n";

  cs::Graph graph(circuit_file);

  tf::Taskflow taskflow;
  tf::Executor executor(num_threads);

  for(auto& node : graph.nodes()) {
    node.set_task_tf(taskflow.emplace([&node, num_itr, length]() {
      node.set_task_tf_status(gpu_mm_cpu_reduce(num_itr, length)); 
    }).name(node.name()));
  }

  // build dependencies
  for(auto& node : graph.nodes()) {
    for(auto fanout : node.fanouts()) {
      node.get_task_tf().precede(fanout->to()->get_task_tf());
    }
  }

  executor.run(taskflow).wait();

  for(auto& node : graph.nodes()) {
    if (!node.get_task_tf_status()) {
      std::cerr << "❌ calculation is wrong!\n";
      std::exit(EXIT_FAILURE);
    }
  }

  std::cout << "✅ all calculations correct\n";

  taskflow.dump(std::cout);

  return 0;
}












