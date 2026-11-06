pybind11_add_module(torontonian SHARED ${CMAKE_CURRENT_SOURCE_DIR}/torontonian.cpp)
pybind11_add_module(pfaffian SHARED ${CMAKE_CURRENT_SOURCE_DIR}/pfaffian.cpp)
pybind11_add_module(permanent SHARED ${CMAKE_CURRENT_SOURCE_DIR}/permanent.cpp)
pybind11_add_module(hafnian SHARED ${CMAKE_CURRENT_SOURCE_DIR}/hafnian.cpp)

target_link_libraries(torontonian PUBLIC torontonianboost looptorontonianboost)

target_link_libraries(pfaffian PUBLIC pfaffianboost)

target_link_libraries(permanent PUBLIC permanentboost permanentlaplaceboost)

target_link_libraries(hafnian PUBLIC hafnianboost)

install(
    TARGETS torontonian pfaffian permanent hafnian
    RUNTIME DESTINATION piquasso/_math
    LIBRARY DESTINATION piquasso/_math
)
