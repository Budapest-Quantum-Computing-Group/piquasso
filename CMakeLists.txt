cmake_minimum_required(VERSION 3.15...3.26)
project(piquasso LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

message(LDFLAGS="$ENV{LDFLAGS}")
message(CXXFLAGS="$ENV{CXXFLAGS}")

if(MSVC)
  add_compile_options(/W4 /WX /openmp)
else()
  add_compile_options(
  $<$<COMPILE_LANGUAGE:CXX>:-Wall>
  $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
  $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
  $<$<COMPILE_LANGUAGE:CXX>:-Werror>
  )
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CUDA support only on Linux x86_64 / aarch64 (JAX does not support CUDA on Windows or macOS)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
   (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64"))
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
    endif()
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG)
endif()


add_subdirectory(src)
add_subdirectory(piquasso/_math)
add_subdirectory(piquasso/jax_extensions)
