#ifndef KERNELS_H_
#define KERNELS_H_

#include "xla/ffi/api/ffi.h"
#include <cuda_runtime_api.h>

namespace ffi = xla::ffi;

ffi::Error PermImpl(cudaStream_t stream, ffi::Buffer<ffi::C128> A,
                    ffi::Buffer<ffi::U64> rows,
                    ffi::Buffer<ffi::U64> cols,
                    ffi::ResultBuffer<ffi::C128> permanent);

ffi::Error PermFwdImpl(cudaStream_t stream, ffi::Buffer<ffi::C128> A, ffi::Buffer<ffi::U64> rows,
                       ffi::Buffer<ffi::U64> cols,
                       ffi::ResultBuffer<ffi::C128> y,
                       ffi::ResultBuffer<ffi::C128> res);

ffi::Error PermBwdImpl(cudaStream_t stream, ffi::Buffer<ffi::C128> res, ffi::Buffer<ffi::C128> A,
                       ffi::Buffer<ffi::U64> rows, ffi::Buffer<ffi::U64> cols,
                       ffi::Buffer<ffi::C128> cotangent,
                       ffi::ResultBuffer<ffi::C128> ct_x);

#endif // KERNELS_H_