#pragma once

#include <numeric>
#include "coro_scheduler_base.hpp"
#include "central_queue.hpp"
#include "central_priority_queue.hpp"

namespace cs {

  using Scheduler = SchedulerCentralQueue;  // Default alias users can override if needed

}
