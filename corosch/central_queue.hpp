#pragma once
#include <coroutine>
#include <list>
#include <queue>
#include <vector>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <memory>
#include <string>
#include <iostream>
#include "task.hpp"

namespace cs {

class SchedulerCentralQueue : public CoroScheduler {

public: 
  SchedulerCentralQueue(size_t num_threads = std::thread::hardware_concurrency());

  Task* emplace(std::coroutine_handle<Task::promise_type> h);
  void schedule();
  void wait();
  auto suspend();

private:
  std::vector<std::unique_ptr<Task>> _tasks;
  std::queue<Task*> _pending_tasks;
  std::vector<std::thread> _workers;
  std::mutex _mtx;
  std::condition_variable _cv;
  bool _stop{false};
  std::atomic<size_t> _finished{0};

  void _enqueue(Task* task);
  void _process(Task* task);
};

// Constructor spawns worker threads
inline
SchedulerCentralQueue::SchedulerCentralQueue(size_t num_threads) {
  _workers.reserve(num_threads);
  for(size_t t = 0; t < num_threads; ++t) {
    _workers.emplace_back([this]() {
      while(true) {
        Task* task = nullptr;
        {
          std::unique_lock<std::mutex> lock(_mtx);
          _cv.wait(lock, [this]{ return _stop || !_pending_tasks.empty(); });
          if(_stop) return;
          task = _pending_tasks.front();
          _pending_tasks.pop();
        }
        if(task) {
          _process(task);
        }
      }
    });
  }
}

inline
Task* SchedulerCentralQueue::emplace(std::coroutine_handle<Task::promise_type> h) {
  _tasks.emplace_back(std::make_unique<Task>(h));
  return _tasks.back().get();
}

// Schedule all ready tasks
inline
void SchedulerCentralQueue::schedule() {

  std::vector<Task*> srcs;

  for(auto& task : _tasks) {
    if(task->dependency_count.load(std::memory_order_relaxed) == 0) {
      // std::cerr << "source enqueue task: " << task->name << "\n";
      srcs.push_back(task.get()); 
    }
  }

  for(auto& task : srcs) {
    _enqueue(task);
  }
}

inline
auto SchedulerCentralQueue::suspend() {
  return std::suspend_always{};
}

inline
void SchedulerCentralQueue::wait() {
  for(auto& w : _workers) {
    w.join();
  }
}

// Enqueue task
inline
void SchedulerCentralQueue::_enqueue(Task* task) {
  {
    std::unique_lock<std::mutex> lock(_mtx);
    _pending_tasks.push(task);
  }
  _cv.notify_one();
}

// Run task, handle dependencies
inline
void SchedulerCentralQueue::_process(Task* node) {

  // this is where the problem exist, some nodes have been finished but still got resumed?
  // if (node->finished.exchange(true)) {
  //   // Already finished or being processed — skip
  //   return;
  // }

  // std::cerr << "processing task: " << node->name << "\n";

  node->handle.resume();

  // std::cerr << "resume back for task: " << node->name << "\n";

  if(!node->handle.done()) {
    // std::cerr << "anything happen here for task: " << node->name << "\n";
    _enqueue(node);
  } else {
    node->handle.destroy();

    for(auto succ : node->successors) {
      if(succ->dependency_count.fetch_sub(1) == 1) {
        // std::cerr << "enqueue suc task: " << succ->name << "\n";
        _enqueue(succ);
      }
    }

    if(_finished.fetch_add(1) + 1 == _tasks.size()) {
      {
        std::unique_lock<std::mutex> lock(_mtx);
        _stop = true;
      }
      _cv.notify_all();
    }
  }
}

} // end of namespace cs

