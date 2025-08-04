#pragma once
#include "task.hpp"

namespace cs {

class CoroScheduler {
public:
  virtual ~CoroScheduler() = default;
  virtual Task* emplace(std::coroutine_handle<Task::promise_type> h) = 0;
  virtual void schedule() = 0;
  virtual void wait() = 0;
};

} // namespace cs 

