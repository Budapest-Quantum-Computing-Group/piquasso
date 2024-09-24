pybind11_add_module(torontonian SHARED ${CMAKE_CURRENT_SOURCE_DIR}/torontonian.cpp)

target_link_libraries(torontonian PUBLIC torontonianboost looptorontonianboost)

install(TARGETS torontonian LIBRARY DESTINATION piquasso/_math)
