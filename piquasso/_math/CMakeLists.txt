pybind11_add_module(torontonian SHARED ${CMAKE_CURRENT_SOURCE_DIR}/torontonian.cpp)
pybind11_add_module(pfaffian SHARED ${CMAKE_CURRENT_SOURCE_DIR}/pfaffian.cpp)

target_link_libraries(torontonian PUBLIC torontonianboost looptorontonianboost)

target_link_libraries(pfaffian PUBLIC pfaffianboost)

install(
    TARGETS torontonian pfaffian
    RUNTIME DESTINATION piquasso/_math
    LIBRARY DESTINATION piquasso/_math
)