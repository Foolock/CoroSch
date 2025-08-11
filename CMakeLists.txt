cmake_minimum_required(VERSION 3.26)  # Ensure you have at least CMake 3.8
project(CoroSch VERSION 1.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# compile and link CUDA code separately from the rest of your code to
# speed up the build process and reduce the size of your final executable.
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall -O3 --extended-lambda -Xcompiler -fcoroutines")

# set compilation flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fcoroutines")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to Release ")
  set(
    CMAKE_BUILD_TYPE Release
    CACHE
    STRING "Choose the type of build."
    FORCE
  )
  # Set the possible values of build type for cmake-gui
  set_property(
    CACHE
    CMAKE_BUILD_TYPE
    PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo"
  )
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

set(COROSCH_3RD_PARTY_DIR ${PROJECT_SOURCE_DIR}/3rd-party)
set(COROSCH_EXAMPLE_DIR ${PROJECT_SOURCE_DIR}/examples)

# -----------------------------------------------------------------------------
# must-have package include
# -----------------------------------------------------------------------------
include(CheckLanguage)
include_directories(/usr/local/cuda/include)

# Enable test
include(CTest)
set(CMAKE_CTEST_ARGUMENTS "--output-on-failure")

add_subdirectory(examples)

message(STATUS "COROSCH_3RD_PARTY_DIR = ${COROSCH_3RD_PARTY_DIR}")
message(STATUS "COROSCH_EXAMPLE_DIR = ${COROSCH_EXAMPLE_DIR}")

