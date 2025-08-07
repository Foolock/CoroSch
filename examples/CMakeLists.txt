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

list(APPEND COROSCH_EXAMPLES
  lc_tf
  lc_central_queue
  lc_central_priority_queue
  fd_tf
  fd_central_queue
  fd_central_priority_queue
  lc_taro
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


