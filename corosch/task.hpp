// task.hpp
#pragma once
#include <coroutine>
#include <vector>
#include <atomic>
#include <string>

namespace cs {

struct Task {
  std::coroutine_handle<> handle;
  std::vector<Task*> successors;
  std::atomic<size_t> dependency_count{0};
  std::atomic<size_t> resume_count{0};
  std::string name;
  std::atomic<bool> finished{false};
  std::atomic<size_t> push_count{0};
  std::atomic<size_t> pop_count{0};

  struct promise_type {
    bool done = false;
    std::string name;

    std::suspend_always initial_suspend() noexcept { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }

    Task get_return_object() {
      return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
    }
    // void return_void() {}
    void return_value(bool d) { done = d; }
    void unhandled_exception() {}
  };

  Task(std::coroutine_handle<promise_type> h) : handle(h) {}

  Task* set_name(const std::string& name) {
    this->name = name;
    this->get_handle().promise().name = name;
    return this;
  }

  inline bool is_done() {
    return get_handle().promise().done;
  }

  void precede(Task* next) {
    successors.push_back(next);
    next->dependency_count.fetch_add(1, std::memory_order_relaxed);
  }

  std::coroutine_handle<promise_type> get_handle() { 
    return std::coroutine_handle<promise_type>::from_address(handle.address()); 
  }
};

} // namespace cs 

