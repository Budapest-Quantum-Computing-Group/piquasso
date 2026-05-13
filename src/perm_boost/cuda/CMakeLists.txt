# CUDA build for the perm_boost JAX extension.
#
# Expected inputs from the caller:
#   JAX_INCLUDE_DIR  -- path to XLA/JAX FFI headers
#
# Skipped silently if CUDA was not found at the top level.

if(NOT CUDAToolkit_FOUND)
    return()
endif()

# Prevent pybind11 from injecting -flto: with CUDA_SEPARABLE_COMPILATION, LTO
# causes the device-link step to emit GCC IR instead of native code, leaving
# fatbinData undefined at runtime. Setting this before pybind11_add_module
# suppresses pybind11::lto entirely.
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

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

set_target_properties(_perm_boost_gpu_ops PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    # pybind11_add_module defaults to hidden visibility; with -rdc=true nvcc
    # makes fatbinData a LOCAL symbol that the device-link step cannot resolve.
    # Override to default so fat-binary registration symbols remain globally
    # visible.
    CUDA_VISIBILITY_PRESET default
    # SASS+PTX for Pascal through Hopper; PTX forward-JITs to Blackwell+.
    # Maxwell (sm_5x) is excluded.
    CUDA_ARCHITECTURES "60;70;75;80;86;89;90"
)

install(
    TARGETS _perm_boost_gpu_ops
    RUNTIME DESTINATION piquasso/jax_extensions
    LIBRARY DESTINATION piquasso/jax_extensions
)
