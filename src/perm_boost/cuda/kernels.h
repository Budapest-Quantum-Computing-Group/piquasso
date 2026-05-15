/*
 * Copyright 2021-2026 Budapest Quantum Computing Group
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Portions of this file are based on work by Bence Soóki-Tóth, used with
 * permission and originally made available under the MIT License.
 *
 * Bence Soóki-Tóth. "Efficient calculation of permanent function gradients
 * in photonic quantum computing simulations", Eötvös Loránd University, 2025.
 */

#ifndef KERNELS_H_
#define KERNELS_H_

#include "xla/ffi/api/ffi.h"
#include <cuda_runtime_api.h>

namespace ffi = xla::ffi;

ffi::Error PermImpl(cudaStream_t stream, ffi::Buffer<ffi::DataType::C128> A,
                    ffi::Buffer<ffi::DataType::U64> rows,
                    ffi::Buffer<ffi::DataType::U64> cols,
                    ffi::ResultBuffer<ffi::DataType::C128> permanent);

ffi::Error PermFwdImpl(cudaStream_t stream, ffi::Buffer<ffi::DataType::C128> A, ffi::Buffer<ffi::DataType::U64> rows,
                       ffi::Buffer<ffi::DataType::U64> cols,
                       ffi::ResultBuffer<ffi::DataType::C128> y,
                       ffi::ResultBuffer<ffi::DataType::C128> res);

ffi::Error PermBwdImpl(cudaStream_t stream, ffi::Buffer<ffi::DataType::C128> res, ffi::Buffer<ffi::DataType::C128> A,
                       ffi::Buffer<ffi::DataType::U64> rows, ffi::Buffer<ffi::DataType::U64> cols,
                       ffi::Buffer<ffi::DataType::C128> cotangent,
                       ffi::ResultBuffer<ffi::DataType::C128> ct_x);

#endif // KERNELS_H_
