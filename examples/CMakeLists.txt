set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/examples)

add_executable(linear_chain_tf ${COROSCH_EXAMPLE_DIR}/linear_chain_tf.cu)

list(APPEND COROSCH_EXAMPLES
  linear_chain_tf
)

foreach(example IN LISTS COROSCH_EXAMPLES) 
  # include directories
  target_include_directories(${example}
    PUBLIC 
    ${COROSCH_3RD_PARTY_DIR}/taskflow
    ${PROJECT_SOURCE_DIR}/corosch)

endforeach()


