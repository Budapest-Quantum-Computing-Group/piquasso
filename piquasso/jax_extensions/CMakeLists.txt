# Build C++/CUDA JAX extensions exposed via piquasso.jax_extensions.*
#
# This subdir is only useful when JAX is importable in the build env (its
# headers ship the XLA FFI public API). When JAX is unavailable we skip
# silently -- the resulting wheel is leaner but still functional, and at
# runtime the dispatchers fall back to the JAX-only paths.

# Locate the XLA/JAX FFI include directory. Callers may pre-set JAX_INCLUDE_DIR
# (e.g. `cmake -DJAX_INCLUDE_DIR=...`) to bypass the Python probe entirely --
# useful for cross-compilation, conda-build, and CI images where the build-env
# Python may not match the runtime environment. Otherwise we query the build-env
# Python, trying jax.ffi (>= 0.6) then jax.extend.ffi (< 0.6) in one call.
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

# Validate that the include dir actually contains the FFI header. Catches slim
# wheels, stripped installs, or a mis-set -DJAX_INCLUDE_DIR before we hit an
# opaque compile error later.
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

# Warning suppressions for permanent.hpp / perm_boost_core.cpp.
#
# Piquasso's root CMakeLists.txt sets -Wall -Wextra -Wpedantic -Werror (and
# /W4 /WX on MSVC). These suppressions are scoped to this target only -- they
# do NOT relax the policy for the rest of the project.
#
# GCC/Clang:
#   -Wno-pedantic         permanent.hpp uses __int128, a GCC extension
#   -Wno-unknown-pragmas  #pragma omp under compilers without OpenMP enabled
#   -Wno-unused-variable
#   -Wno-unused-parameter pybind11 binding glue declares args it doesn't read
#   -Wno-sign-compare     loop counters cross signed/unsigned in the kernel
#
# MSVC (rough parallels to the above):
#   /wd4002 /wd4003       macro-arity warnings from the permanent macros
#   /wd4456               local-shadowing in the inlined kernel
#   /wd4715               not-all-paths-return in templated specialisations
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

# Discovery-first OpenMP. find_package handles Linux, Windows, and any Apple
# setup with a CMake-aware toolchain (e.g. a brewed GCC, or a user-supplied
# OpenMP_ROOT / CMAKE_PREFIX_PATH). Only if that fails on Apple do we inject
# hints for the common Apple-Clang + libomp arrangements -- ordered so the
# first one that has both headers and the dylib wins.
find_package(OpenMP)

if(NOT OpenMP_CXX_FOUND AND APPLE)
    foreach(_prefix
        "/opt/homebrew/opt/libomp"   # Homebrew on Apple Silicon
        "/usr/local/opt/libomp"      # Homebrew on Intel
        "/opt/local")                # MacPorts (libomp port)
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

# GPU module: only built when CUDA was enabled at the root level. The subdir
# self-guards on CUDAToolkit_FOUND, but we also gate the add_subdirectory call
# itself so configuration never traverses into the CUDA tree on systems
# without a toolkit. JAX_INCLUDE_DIR (set above) is consumed inside.
if(CUDAToolkit_FOUND)
    add_subdirectory(
        ${PERM_BOOST_SRC}/cuda
        ${CMAKE_BINARY_DIR}/src/perm_boost/cuda
    )
endif()
