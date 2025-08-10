#include "graph.hpp"
#include "kernels.cuh"

cs::Task cpu_mm_reduce_coro(cs::SchedulerWorkStealing& sch, const int num_itr, const int length, cs::Node* node, int& finished) {

  finished++;

  fprintf(stderr, "begin CPU task %s, finished = %d\n", node->name().c_str(), finished);
  // std::cerr << "begin CPU task: " << node->_name << "\n";

  std::vector<float> A(length * length, 1.0f);
  std::vector<float> B(length * length, 1.0f);
  std::vector<float> C(length * length, 0.0f);

  float cpu_sum = 0.0f;

  for (int itr = 0; itr < num_itr; ++itr) {
    // Matrix multiplication: C = A * B
    for (int i = 0; i < length; ++i) {
      for (int j = 0; j < length; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < length; ++k) {
          sum += A[i * length + k] * B[k * length + j];
        }
        C[i * length + j] = sum;
      }
    }

    // Optional: allow coroutine scheduler to interleave (not strictly needed unless yielding is desired)
    // co_await sch.suspend();

    // CPU reduction
    float partial_sum = std::accumulate(C.begin(), C.end(), 0.0f);
    cpu_sum += partial_sum;
  }

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

  cs::SchedulerWorkStealing coro_scheduler(num_threads);

  int finished = 0;

  // emplace tasks
  for(auto& node : graph.nodes()) {
    node.set_task_coro(coro_scheduler.emplace(
      cpu_mm_reduce_coro(coro_scheduler, num_itr, length, &node, finished).get_handle()
    )->set_name(node.name())); 
  }

  // build dependencies
  for(auto& node : graph.nodes()) {
    for(auto fanout : node.fanouts()) {
      node.get_task_coro()->precede(fanout->to()->get_task_coro());
    }
  }

  coro_scheduler.schedule();
  coro_scheduler.wait();

  for(auto& node : graph.nodes()) {
    if (!node.get_task_coro()->is_done()) {
      std::cerr << "❌ calculation is wrong!\n";
      std::exit(EXIT_FAILURE);
    }
  }

  std::cout << "✅ all calculations correct\n";


  return 0;
}












