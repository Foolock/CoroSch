set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/examples)

message(STATUS "HERE, COROSCH_3RD_PARTY_DIR = ${COROSCH_3RD_PARTY_DIR}")

# CUDA kernels
add_library(kernels STATIC kernels.cu)
set_target_properties(kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(kernels PROPERTIES CUDA_ARCHITECTURES 86)

add_executable(lc_tf ${COROSCH_EXAMPLE_DIR}/lc_tf.cpp)
add_executable(lc_central_queue ${COROSCH_EXAMPLE_DIR}/lc_central_queue.cpp)
add_executable(lc_central_priority_queue ${COROSCH_EXAMPLE_DIR}/lc_central_priority_queue.cpp)
add_executable(fd_tf ${COROSCH_EXAMPLE_DIR}/fd_tf.cpp)
add_executable(fd_central_queue ${COROSCH_EXAMPLE_DIR}/fd_central_queue.cpp)
add_executable(fd_central_priority_queue ${COROSCH_EXAMPLE_DIR}/fd_central_priority_queue.cpp)
add_executable(lc_taro ${COROSCH_EXAMPLE_DIR}/lc_taro.cu)
add_executable(fd_taro ${COROSCH_EXAMPLE_DIR}/fd_taro.cu)
add_executable(fd_work_stealing ${COROSCH_EXAMPLE_DIR}/fd_work_stealing.cpp)
add_executable(run_circuit_coro ${COROSCH_EXAMPLE_DIR}/run_circuit_coro.cpp)
add_executable(run_circuit_tf ${COROSCH_EXAMPLE_DIR}/run_circuit_tf.cpp)
add_executable(run_circuit_non_coro ${COROSCH_EXAMPLE_DIR}/run_circuit_non_coro.cpp)

list(APPEND COROSCH_EXAMPLES
  lc_tf
  lc_central_queue
  lc_central_priority_queue
  fd_tf
  fd_central_queue
  fd_central_priority_queue
  lc_taro
  fd_taro
  fd_work_stealing
  run_circuit_coro
  run_circuit_tf
  run_circuit_non_coro
)

foreach(example IN LISTS COROSCH_EXAMPLES) 
  # include directories
  target_include_directories(${example}
    PUBLIC 
    ${COROSCH_3RD_PARTY_DIR}/taskflow
    ${COROSCH_3RD_PARTY_DIR}/taro
    ${PROJECT_SOURCE_DIR}/corosch)

  target_link_libraries(${example} PRIVATE kernels)

  target_compile_options(${example} PRIVATE -fcoroutines)

endforeach()


