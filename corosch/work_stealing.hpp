#pragma once
#include <coroutine>
#include <vector>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <memory>
#include <string>
#include <iostream>
#include "task.hpp"
#include "wsq.hpp" // Your implementation of WorkStealingQueue<T>

namespace cs {

class SchedulerWorkStealing : public CoroScheduler {

public: 
  SchedulerWorkStealing(size_t num_threads = std::thread::hardware_concurrency());

  Task* emplace(std::coroutine_handle<Task::promise_type> h);
  void schedule();
  void wait();
  auto suspend();

private:
  std::vector<std::unique_ptr<Task>> _tasks;
  std::vector<std::thread> _workers;
  std::vector<std::unique_ptr<WorkStealingQueue<Task*>>> _queues;
  std::atomic<size_t> _finished{0};
  std::atomic<bool> _stop{false};

  void _enqueue(Task* task, size_t tid);
  void _process(Task* node, size_t tid);
};

// Constructor: create queues and spawn worker threads
inline
SchedulerWorkStealing::SchedulerWorkStealing(size_t num_threads) {
  _queues.resize(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    _queues[i] = std::make_unique<WorkStealingQueue<Task*>>();
  }

  _workers.reserve(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    _workers.emplace_back([this, i, num_threads]() {
      while (!_stop.load(std::memory_order_acquire)) {
        
        std::optional<Task*> opt;
        Task* task = nullptr;
        
        // first process its own queue
        while(!_queues[i]->empty()) {
          opt = _queues[i]->pop();
          if(opt.has_value()) {
            task = opt.value();
            if(task) {
              _process(task, i);
            }
          }
        }

        // then steal from others' queues
        size_t to_steal = 0;
        for(size_t j=0; j<num_threads; j++) { // j = steal_from(last_steal) 
          if(j == i) {
            continue; 
          }
          opt = _queues[j]->steal();
          if(opt.has_value()) {
            to_steal = j;
            break;
          }
        }
        if(!opt.has_value()) {
          continue;
        }
        task = opt.value(); 
        if(task) {
          _process(task, to_steal);
        }
      }
    });
  }
}

inline
Task* SchedulerWorkStealing::emplace(std::coroutine_handle<Task::promise_type> h) {
  _tasks.emplace_back(std::make_unique<Task>(h));
  return _tasks.back().get();
}

inline
void SchedulerWorkStealing::schedule() {

  std::vector<Task*> srcs;

  for(auto& task : _tasks) {
    if(task->dependency_count.load(std::memory_order_relaxed) == 0) {
      // std::cerr << "source enqueue task: " << task->name << "\n";
      srcs.push_back(task.get()); 
    }
  }

  // put all source tasks in the first wsq
  for(auto& task : srcs) {
    _enqueue(task, 0);
  }
}

inline
auto SchedulerWorkStealing::suspend() {
  return std::suspend_always{};
}

inline
void SchedulerWorkStealing::wait() {
  for (auto& w : _workers) {
    w.join();
  }
}

inline
void SchedulerWorkStealing::_enqueue(Task* task, size_t tid) {
  _queues[tid]->push(task);
}

inline
void SchedulerWorkStealing::_process(Task* node, size_t tid) {

  node->handle.resume();

  if (!node->handle.done()) {
    _enqueue(node, tid);
  } else {

    node->handle.destroy();

    node->finished.store(true);

    // put successors into my own queue and let others steal them 
    for (auto& succ : node->successors) {
      if (succ->dependency_count.fetch_sub(1) == 1) {
        _enqueue(succ, tid);
      }
    }

    if (_finished.fetch_add(1) + 1 == _tasks.size()) {
      _stop = true;
    }
  }
}

}  // namespace Coro

