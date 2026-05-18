# CUDA build for the perm_boost JAX extension.
#
# Expected inputs from the caller:
#   JAX_INCLUDE_DIR  -- path to XLA/JAX FFI headers
#
# Skipped silently if CUDA was not found at the top level.

if(NOT CUDAToolkit_FOUND)
    return()
endif()

pybind11_add_module(_perm_boost_gpu_ops SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels.cc.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_ops.cc
)

target_include_directories(_perm_boost_gpu_ops PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)
target_include_directories(_perm_boost_gpu_ops SYSTEM PRIVATE ${JAX_INCLUDE_DIR})

# Static CUDA runtime: makes the resulting .so self-contained so the user
# does not need a system CUDA install at runtime -- only an NVIDIA driver
# compatible with CUDA 12.
target_link_libraries(_perm_boost_gpu_ops PRIVATE CUDA::cudart_static)

target_compile_options(_perm_boost_gpu_ops PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-pedantic -Wno-unknown-pragmas>
)

install(
    TARGETS _perm_boost_gpu_ops
    RUNTIME DESTINATION piquasso/jax_extensions
    LIBRARY DESTINATION piquasso/jax_extensions
)
