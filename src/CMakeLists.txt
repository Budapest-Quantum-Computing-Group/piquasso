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

target_include_directories(torontonianboost PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(looptorontonianboost PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
