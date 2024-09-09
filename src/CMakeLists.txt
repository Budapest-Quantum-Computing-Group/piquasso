add_library(
    torontonianboost
    STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/torontonian.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/torontonian.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matrix.hpp
)

target_include_directories(torontonianboost PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})