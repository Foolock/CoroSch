#pragma once
#include <vector>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <memory>
#include <string>
#include <iostream>
#include <functional>
#include "wsq.hpp" // Your implementation of WorkStealingQueue<T>

namespace cs {

// normal task
struct NTask {
  std::function<void()> func;
  std::vector<NTask*> successors;
  std::atomic<size_t> dependency_count{0};
  std::string name;
  std::atomic<bool> finished{false};
  std::atomic<size_t> push_count{0};
  std::atomic<size_t> pop_count{0};

  NTask(std::function<void()> f, const std::string& n = "")
      : func(std::move(f)), name(n) {}
};

class SchedulerWorkStealingN {

public:
  SchedulerWorkStealingN(size_t num_threads = std::thread::hardware_concurrency());

  cs::NTask* emplace(std::function<void()> func, const std::string& name = "");
  void schedule();
  void wait();

private:
  std::vector<std::unique_ptr<cs::NTask>> _tasks;
  std::vector<std::thread> _workers;
  std::vector<std::unique_ptr<WorkStealingQueue<cs::NTask*>>> _queues;
  std::atomic<size_t> _finished{0};
  std::atomic<bool> _stop{false};

  void _enqueue(cs::NTask* task, size_t tid);
  void _process(cs::NTask* node, size_t tid);
};

// Constructor
inline
SchedulerWorkStealingN::SchedulerWorkStealingN(size_t num_threads) {
  _queues.resize(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    _queues[i] = std::make_unique<WorkStealingQueue<cs::NTask*>>();
  }

  _workers.reserve(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    _workers.emplace_back([this, i, num_threads]() {
      while (!_stop.load(std::memory_order_acquire)) {

        std::optional<cs::NTask*> opt;
        cs::NTask* task = nullptr;

        // process own queue
        while(!_queues[i]->empty()) {
          opt = _queues[i]->pop();
          if(opt.has_value()) {
            task = opt.value();
            if(task) {
              task->pop_count.fetch_add(1);
              _process(task, i);
            }
          }
        }

        fprintf(stderr, "running loop...\n");

        // steal from others
        // for(size_t j=0; j<num_threads; ++j) {
        //   if(j == i) continue;
        //   opt = _queues[j]->steal();
        //   if(opt.has_value()) {
        //     task = opt.value();
        //     if(task) {
        //       _process(task, j);
        //     }
        //     break;
        //   }
        // }
      }
    });
  }
}

inline
cs::NTask* SchedulerWorkStealingN::emplace(std::function<void()> func, const std::string& name) {
  _tasks.emplace_back(std::make_unique<cs::NTask>(std::move(func), name));
  return _tasks.back().get();
}

inline
void SchedulerWorkStealingN::schedule() {
  std::vector<cs::NTask*> srcs;
  for(auto& task : _tasks) {
    if(task->dependency_count.load(std::memory_order_relaxed) == 0) {
      srcs.push_back(task.get());
    }
  }
  for(auto& task : srcs) {
    if(task->finished.load()) {
      fprintf(stderr, "enqueue destroy source %s\n", task->name.c_str());
    }
    _enqueue(task, 0);
  }
}

inline
void SchedulerWorkStealingN::wait() {
  for (auto& w : _workers) {
    w.join();
  }
}

inline
void SchedulerWorkStealingN::_enqueue(cs::NTask* task, size_t tid) {
  _queues[tid]->push(task);
  task->push_count.fetch_add(1);
}

inline
void SchedulerWorkStealingN::_process(cs::NTask* node, size_t tid) {
  fprintf(stderr, "run CPU task %s, push = %ld, pop = %ld\n",
          node->name.c_str(),
          node->push_count.load(),
          node->pop_count.load());

  node->func();
  node->finished.store(true);

  for (auto& succ : node->successors) {
    if (succ->dependency_count.fetch_sub(1) == 1) {
      if(succ->finished.load()) {
        fprintf(stderr, "enqueue destroy succ %s\n", succ->name.c_str());
      }
      _enqueue(succ, tid);
    }
  }

  if (_finished.fetch_add(1) + 1 == _tasks.size()) {
    _stop = true;
  }

  fprintf(stderr, "finish task %s, _finished = %ld\n",
          node->name.c_str(), _finished.load());
}

} // end of namespace cs
