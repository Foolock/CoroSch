set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/examples)

message(STATUS "HERE, COROSCH_3RD_PARTY_DIR = ${COROSCH_3RD_PARTY_DIR}")

add_executable(lc_tf ${COROSCH_EXAMPLE_DIR}/lc_tf.cu)
add_executable(lc_central_queue ${COROSCH_EXAMPLE_DIR}/lc_central_queue.cu)

list(APPEND COROSCH_EXAMPLES
  lc_tf
  lc_central_queue
)

foreach(example IN LISTS COROSCH_EXAMPLES) 
  # include directories
  target_include_directories(${example}
    PUBLIC 
    ${COROSCH_3RD_PARTY_DIR}/taskflow
    ${PROJECT_SOURCE_DIR}/corosch)

endforeach()


