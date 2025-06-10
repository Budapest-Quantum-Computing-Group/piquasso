add_library(
    torontonianboost
    STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/torontonian.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/torontonian_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/torontonian.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/torontonian_common.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matrix.hpp
)

add_library(
    looptorontonianboost
    STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/loop_torontonian.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/torontonian_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/loop_torontonian.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/torontonian_common.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matrix.hpp
)

add_library(
    pfaffianboost
    STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/pfaffian.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pfaffian.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matrix.hpp
)

add_library(
    permanentboost
    STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/permanent.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/permanent.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/n_aryGrayCodeCounter.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.hpp
)

target_include_directories(torontonianboost PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(looptorontonianboost PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(pfaffianboost PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(permanentboost PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


# Source: https://cliutils.gitlab.io/modern-cmake/chapters/packages/OpenMP.html
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(permanentboost PUBLIC OpenMP::OpenMP_CXX)
else()
    # This could happen, because Apple disabled openmp in clang.
    # One could install it via brew, but it would only works on the machine where it
    # is compiled, see:
    # https://cibuildwheel.pypa.io/en/stable/faq/#macos-library-dependencies-do-not-satisfy-target-macos
    # One can download binaries from here:
    # https://mac.r-project.org/openmp/
    # If one would like to make this work, one would need to download the correct binary
    # and compile the project for every macOS version. We will avoid this, and build on
    # macOS without OpenMP.
    message("OpenMP not found, building without it.")
endif()
