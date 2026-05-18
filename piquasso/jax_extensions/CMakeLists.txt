# Build C++/CUDA JAX extensions exposed via piquasso.jax_extensions.*
#
# This subdir is only useful when JAX is importable in the build env (its
# headers ship the XLA FFI public API). When JAX is unavailable we skip
# silently -- the resulting wheel is leaner but still functional, and at
# runtime the dispatchers fall back to the JAX-only paths.

if(NOT DEFINED JAX_INCLUDE_DIR)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "
try:
    import jax.ffi as _f
except ImportError:
    import jax.extend.ffi as _f
print(_f.include_dir())
"
    OUTPUT_VARIABLE JAX_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE JAX_RESULT
    ERROR_QUIET
  )
  if(NOT JAX_RESULT EQUAL 0)
    message(STATUS "JAX not found in build env, skipping jax_extensions build")
    return()
  endif()
endif()

if(NOT EXISTS "${JAX_INCLUDE_DIR}/xla/ffi/api/ffi.h")
  message(STATUS
    "JAX_INCLUDE_DIR=${JAX_INCLUDE_DIR} is missing xla/ffi/api/ffi.h; "
    "skipping jax_extensions build")
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

# permanent.hpp uses __int128 (GCC extension) and #pragma omp -- suppress
# pedantic and unknown-pragma warnings that Piquasso's global flags would
# otherwise error on.
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

find_package(OpenMP)

if(NOT OpenMP_CXX_FOUND AND APPLE)
    foreach(_prefix
        "/opt/homebrew/opt/libomp"
        "/usr/local/opt/libomp"
        "/opt/local")
        if(EXISTS "${_prefix}/lib/libomp.dylib" AND
           EXISTS "${_prefix}/include/omp.h")
            set(OpenMP_C_FLAGS
                "-Xpreprocessor -fopenmp -I${_prefix}/include"
                CACHE STRING "" FORCE)
            set(OpenMP_CXX_FLAGS
                "-Xpreprocessor -fopenmp -I${_prefix}/include"
                CACHE STRING "" FORCE)
            set(OpenMP_C_LIB_NAMES "omp" CACHE STRING "" FORCE)
            set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)
            set(OpenMP_omp_LIBRARY "${_prefix}/lib/libomp.dylib"
                CACHE FILEPATH "" FORCE)
            break()
        endif()
    endforeach()
    find_package(OpenMP)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(_perm_boost_core PRIVATE OpenMP::OpenMP_CXX)
endif()

# Both RUNTIME and LIBRARY are specified so the install works on Windows
# (.dll = RUNTIME) and Linux/macOS (.so = LIBRARY).
install(
    TARGETS _perm_boost_core
    RUNTIME DESTINATION piquasso/jax_extensions
    LIBRARY DESTINATION piquasso/jax_extensions
)

# GPU module: only built when CUDA was enabled at the root level. The CUDA
# subdir has its own CMakeLists.txt and uses JAX_INCLUDE_DIR set above.
if(CUDAToolkit_FOUND)
    add_subdirectory(
        ${PERM_BOOST_SRC}/cuda
        ${CMAKE_BINARY_DIR}/src/perm_boost/cuda
    )
endif()
