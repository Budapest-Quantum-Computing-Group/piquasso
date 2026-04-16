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
    m.doc() = R"pbdoc(
          Permanent calculator plugin
          -----------------------
  
          .. currentmodule:: scikit_build_example
  
          .. autosummary::
             :toctree: _generate
  
             permanent
      )pbdoc";
    m.def("foo", []()
          {
      py::dict registrations;
      registrations["dperm"] = EncapsulateFfiHandler(Perm);
    registrations["dperm_fwd"] = EncapsulateFfiHandler(PermFwd);
     registrations["dperm_bwd"] = EncapsulateFfiHandler(PermBwd);
      return registrations; });
    m.attr("__version__") = "dev";
}
