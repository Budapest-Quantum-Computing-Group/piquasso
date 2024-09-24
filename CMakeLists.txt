cmake_minimum_required(VERSION 3.15...3.26)
project(piquasso LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

message(LDFLAGS="$ENV{LDFLAGS}")
message(CXXFLAGS="$ENV{CXXFLAGS}")

if(MSVC)
  add_compile_options(/W4 /WX /openmp)
else()
  add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG)
endif()


add_subdirectory(src)
add_subdirectory(piquasso/_math)
