#include "taskflow/taskflow.hpp"

int main(int argc, char *argv[]) {

  if(argc != 5) {
    std::cerr << "usage: ./example/linear_chain_tf num_itr length num_nodes num_threads\n";
    std::exit(EXIT_FAILURE);
  }

  int num_itr = std::atoi(argv[1]);
  int length = std::atoi(argv[2]);
  int num_nodes = std::atoi(argv[3]);
  int num_threads = std::atoi(argv[4]);

  return 0;
}
