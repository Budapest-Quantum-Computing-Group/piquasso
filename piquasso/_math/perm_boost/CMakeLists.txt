# Locate the XLA/JAX FFI include directory by querying the build-env Python.
# JAX is in [build-system].requires so it is always available. Two locations
# are tried: jax.ffi (≥ 0.6) then jax.extend.ffi (< 0.6).
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import jax.ffi; print(jax.ffi.include_dir())"
  OUTPUT_VARIABLE JAX_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE JAX_RESULT
  ERROR_QUIET
)

if(NOT JAX_RESULT EQUAL 0)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import jax.extend.ffi; print(jax.extend.ffi.include_dir())"
    OUTPUT_VARIABLE JAX_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE JAX_RESULT
    ERROR_QUIET
  )
endif()

# perm_boost requires JAX; skip the build gracefully if it is not available.
if(NOT JAX_RESULT EQUAL 0)
  message(STATUS "JAX not found, skipping perm_boost build")
  return()
endif()

set(PERM_BOOST_SRC ${CMAKE_SOURCE_DIR}/src/perm_boost)

# CPU module: _perm_boost_core
pybind11_add_module(_perm_boost_core SHARED
    ${PERM_BOOST_SRC}/perm_boost_core.cpp
    ${CMAKE_SOURCE_DIR}/src/permanent.cpp
)

target_include_directories(_perm_boost_core PRIVATE
    ${PERM_BOOST_SRC}
    ${CMAKE_SOURCE_DIR}/src
)
# Mark XLA/JAX headers as SYSTEM so their warnings are suppressed
target_include_directories(_perm_boost_core SYSTEM PRIVATE ${JAX_INCLUDE_DIR})

target_compile_definitions(_perm_boost_core PRIVATE
    VERSION_INFO="${PROJECT_VERSION}"
)

# permanent.hpp uses __int128 (GCC extension) and #pragma omp — suppress pedantic
# and unknown-pragma warnings that Piquasso's global flags would otherwise error on
target_compile_options(_perm_boost_core PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wno-pedantic>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wno-unknown-pragmas>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wno-unused-variable>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wno-unused-parameter>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wno-sign-compare>
    $<$<CXX_COMPILER_ID:MSVC>:/wd4002>
    $<$<CXX_COMPILER_ID:MSVC>:/wd4003>
    $<$<CXX_COMPILER_ID:MSVC>:/wd4456>
    $<$<CXX_COMPILER_ID:MSVC>:/wd4715>
)

if(APPLE)
    # Check if Homebrew's libomp is installed
    if(EXISTS "/opt/homebrew/opt/libomp")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    elseif(EXISTS "/usr/local/opt/libomp")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib")
    endif()
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_perm_boost_core PRIVATE OpenMP::OpenMP_CXX)
endif()

# Both RUNTIME and LIBRARY are specified so the install works on Windows (.dll = RUNTIME) and Linux/macOS (.so = LIBRARY).
install(
    TARGETS _perm_boost_core
    RUNTIME DESTINATION piquasso/_math/perm_boost
    LIBRARY DESTINATION piquasso/_math/perm_boost
)

# GPU module: only if CUDA was enabled at the root level
if(CUDAToolkit_FOUND)
    # Prevent pybind11 from injecting -flto: with CUDA_SEPARABLE_COMPILATION, LTO causes the
    # device-link step to emit GCC IR instead of native code, leaving fatbinData undefined at
    # runtime. Setting this before pybind11_add_module suppresses pybind11::lto entirely.
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
    pybind11_add_module(_perm_boost_gpu_ops SHARED
        ${PERM_BOOST_SRC}/kernels.cc.cu
        ${PERM_BOOST_SRC}/gpu_ops.cc
    )
    target_include_directories(_perm_boost_gpu_ops PRIVATE ${PERM_BOOST_SRC})
    target_include_directories(_perm_boost_gpu_ops SYSTEM PRIVATE ${JAX_INCLUDE_DIR})
    # CUDA::cudart provides cuda_runtime_api.h for the host-compiled gpu_ops.cc
    target_link_libraries(_perm_boost_gpu_ops PRIVATE CUDA::cudart)
    target_compile_options(_perm_boost_gpu_ops PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-pedantic -Wno-unknown-pragmas>
    )
    set_target_properties(_perm_boost_gpu_ops PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        # pybind11_add_module defaults to hidden visibility; with -rdc=true nvcc makes
        # fatbinData a LOCAL symbol that the device-link step cannot resolve. Override
        # to default so fat-binary registration symbols remain globally visible.
        CUDA_VISIBILITY_PRESET default
    )
    install(
        TARGETS _perm_boost_gpu_ops
        RUNTIME DESTINATION piquasso/_math/perm_boost
        LIBRARY DESTINATION piquasso/_math/perm_boost
    )
endif()
