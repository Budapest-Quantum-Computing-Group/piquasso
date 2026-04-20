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

#include "kernels.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

XLA_FFI_DEFINE_HANDLER_SYMBOL(Perm, PermImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>() // stream
                                  .Arg<ffi::Buffer<ffi::C128>>()
                                  .Arg<ffi::Buffer<ffi::U64>>()
                                  .Arg<ffi::Buffer<ffi::U64>>()
                                  .Ret<ffi::Buffer<ffi::C128>>(),
                              {xla::ffi::Traits::kCmdBufferCompatible}); // cudaGraph enabled

XLA_FFI_DEFINE_HANDLER_SYMBOL(PermFwd, PermFwdImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>() // stream
                                  .Arg<ffi::Buffer<ffi::C128>>()
                                  .Arg<ffi::Buffer<ffi::U64>>()
                                  .Arg<ffi::Buffer<ffi::U64>>()
                                  .Ret<ffi::Buffer<ffi::C128>>()
                                  .Ret<ffi::Buffer<ffi::C128>>(),
                              {xla::ffi::Traits::kCmdBufferCompatible}); // cudaGraph enabled

XLA_FFI_DEFINE_HANDLER_SYMBOL(PermBwd, PermBwdImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>() // stream
                                  .Arg<ffi::Buffer<ffi::C128>>()            // res
                                  .Arg<ffi::Buffer<ffi::C128>>()            // A
                                  .Arg<ffi::Buffer<ffi::U64>>()             // rows
                                  .Arg<ffi::Buffer<ffi::U64>>()             // cols
                                  .Arg<ffi::Buffer<ffi::C128>>()            // cotangent
                                  .Ret<ffi::Buffer<ffi::C128>>(),           // ct_x
                              {xla::ffi::Traits::kCmdBufferCompatible});    // cudaGraph enabled

template <typename T>
py::capsule EncapsulateFfiHandler(T *fn)
{
    static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                  "Encapsulated function must be and XLA FFI handler");
    return py::capsule(reinterpret_cast<void *>(fn));
}

PYBIND11_MODULE(_perm_boost_gpu_ops, m)
{
    m.doc() = "GPU permanent calculator (XLA FFI, CUDA).";
    m.def("registrations", []()
          {
      py::dict registrations;
      registrations["dperm"] = EncapsulateFfiHandler(Perm);
      registrations["dperm_fwd"] = EncapsulateFfiHandler(PermFwd);
      registrations["dperm_bwd"] = EncapsulateFfiHandler(PermBwd);
      return registrations; });
    m.attr("__version__") = "dev";
}
