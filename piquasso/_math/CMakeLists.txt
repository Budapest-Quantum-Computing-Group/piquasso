pybind11_add_module(torontonian SHARED ${CMAKE_CURRENT_SOURCE_DIR}/torontonian.cpp)
pybind11_add_module(pfaffian SHARED ${CMAKE_CURRENT_SOURCE_DIR}/pfaffian.cpp)
pybind11_add_module(permanent SHARED ${CMAKE_CURRENT_SOURCE_DIR}/permanent.cpp)
pybind11_add_module(
    _hafnian SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/hafnian/_hafnian.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hafnian/hessenberg.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hafnian/labudde.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hafnian/loop_corrections.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hafnian/loop_hafnian.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hafnian/plain_hafnian.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hafnian/powtrace.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hafnian/utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hafnian/workspace.hpp
)
target_include_directories(_hafnian PRIVATE ${PROJECT_SOURCE_DIR}/src)

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_hafnian PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(torontonian PUBLIC torontonianboost looptorontonianboost)

target_link_libraries(pfaffian PUBLIC pfaffianboost)

target_link_libraries(permanent PUBLIC permanentboost permanentlaplaceboost)

install(
    TARGETS torontonian pfaffian permanent
    RUNTIME DESTINATION piquasso/_math
    LIBRARY DESTINATION piquasso/_math
)

install(
    TARGETS _hafnian
    RUNTIME DESTINATION piquasso/_math/hafnian
    LIBRARY DESTINATION piquasso/_math/hafnian
)
