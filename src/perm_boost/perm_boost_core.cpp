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

#include "../matrix.hpp"
#include "../permanent.hpp"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include <algorithm>
#include <complex>
#include <cstdint>
#include <vector>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <type_traits>
#include <utility>


namespace py = pybind11;
namespace ffi = xla::ffi;

template <ffi::DataType T>
std::pair<int64_t, int64_t> get_dims(const ffi::Buffer<T> &buffer)
{
  auto dims = buffer.dimensions();

  if (dims.size() == 0)
  {
    return std::make_pair(0, 0);
  }
  return std::make_pair(static_cast<int64_t>(buffer.element_count()), dims.back());
}

ffi::Error PermanentImpl(ffi::Buffer<ffi::DataType::C128> A, ffi::Buffer<ffi::DataType::U64> rows,
                         ffi::Buffer<ffi::DataType::U64> cols,
                         ffi::ResultBuffer<ffi::DataType::C128> y)
{
  auto [total_size, n] = get_dims(A);

  if (n == 0)
  {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Perm input must be a matrix");
  }
  auto* rows_raw = rows.typed_data();
  auto* cols_raw = cols.typed_data();
  std::vector<int> row_mult(total_size / n);
  std::transform(rows_raw, rows_raw + total_size / n, row_mult.begin(),
                 [](uint64_t v) { return static_cast<int>(v); });
  std::vector<int> col_mult(n);
  std::transform(cols_raw, cols_raw + n, col_mult.begin(),
                 [](uint64_t v) { return static_cast<int>(v); });

  Matrix<std::complex<double>> matrix(total_size / n, n,
                                      A.typed_data());
  Vector<int> row_vec(row_mult.size(), row_mult.data());
  Vector<int> col_vec(col_mult.size(), col_mult.data());

  (*y).typed_data()[0] =
      permanent_cpp<double>(matrix, row_vec, col_vec);

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(Permanent, PermanentImpl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::DataType::C128>>()
                                  .Arg<ffi::Buffer<ffi::DataType::U64>>()
                                  .Arg<ffi::Buffer<ffi::DataType::U64>>()
                                  .Ret<ffi::Buffer<ffi::DataType::C128>>());

ffi::Error PermFwdImpl(ffi::Buffer<ffi::DataType::C128> A, ffi::Buffer<ffi::DataType::U64> rows,
                       ffi::Buffer<ffi::DataType::U64> cols,
                       ffi::ResultBuffer<ffi::DataType::C128> y,
                       ffi::ResultBuffer<ffi::DataType::C128> res)
{
  auto [total_size, n] = get_dims(A);
  if (n == 0)
  {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Permanent input must be a matrix");
  }

  auto* rows_ptr = rows.typed_data();
  auto* cols_ptr = cols.typed_data();
  auto* A_ptr    = A.typed_data();

  std::vector<int> row_mult(total_size / n);
  std::transform(rows_ptr, rows_ptr + total_size / n, row_mult.begin(),
                 [](uint64_t v) { return static_cast<int>(v); });
  std::vector<int> col_mult(n);
  std::transform(cols_ptr, cols_ptr + n, col_mult.begin(),
                 [](uint64_t v) { return static_cast<int>(v); });

  Matrix<std::complex<double>> matrix(total_size / n, n, A_ptr);
  Vector<int> row_vec(row_mult.size(), row_mult.data());
  Vector<int> col_vec(col_mult.size(), col_mult.data());

  (*y).typed_data()[0] =
      permanent_cpp<double>(matrix, row_vec, col_vec);

  // permanent_cpp mutates its matrix and rows arguments; use fresh copies for res
  Matrix<std::complex<double>> matrix2(total_size / n, n, A_ptr);
  Vector<int> row_vec2(row_mult.size(), row_mult.data());
  Vector<int> col_vec2(col_mult.size(), col_mult.data());
  (*res).typed_data()[0] =
      permanent_cpp<double>(matrix2, row_vec2, col_vec2);

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(PermFwd, PermFwdImpl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::DataType::C128>>()
                                  .Arg<ffi::Buffer<ffi::DataType::U64>>()
                                  .Arg<ffi::Buffer<ffi::DataType::U64>>()
                                  .Ret<ffi::Buffer<ffi::DataType::C128>>()
                                  .Ret<ffi::Buffer<ffi::DataType::C128>>());

void ComputePermBwd(Matrix<std::complex<double>> &A,
                    std::vector<int> &rows, std::vector<int> &cols,
                    std::complex<double> cotangent,
                    std::complex<double> *ct_x)
{
  Vector<int> row_vec(rows.size(), rows.data());
  Vector<int> col_vec(cols.size(), cols.data());
  Matrix<std::complex<double>> grad = grad_perm(A, row_vec, col_vec);

  for (int64_t i = 0; i < static_cast<int64_t>(grad.rows); ++i)
  {
    for (int64_t j = 0; j < static_cast<int64_t>(grad.cols); ++j)
    {
      ct_x[i * A.cols + j] = cotangent * grad(i, j);
    }
  }
}

ffi::Error PermBwdImpl([[maybe_unused]] ffi::Buffer<ffi::DataType::C128> res, ffi::Buffer<ffi::DataType::C128> A,
                       ffi::Buffer<ffi::DataType::U64> rows, ffi::Buffer<ffi::DataType::U64> cols,
                       ffi::Buffer<ffi::DataType::C128> cotangent,
                       ffi::ResultBuffer<ffi::DataType::C128> ct_x)
{
  auto A_dims = A.dimensions();
  int64_t ndim = static_cast<int64_t>(A_dims.size());

  if (ndim < 2)
  {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "PermBwd: A must be at least 2D");
  }

  int64_t n_cols = A_dims[ndim - 1];
  int64_t n_rows = A_dims[ndim - 2];
  int64_t batch_size = static_cast<int64_t>(A.element_count()) / (n_rows * n_cols);

  auto* rows_ptr     = rows.typed_data();
  auto* cols_ptr     = cols.typed_data();
  auto* A_ptr        = A.typed_data();
  auto* cot_ptr      = cotangent.typed_data();
  auto* ct_x_ptr     = (*ct_x).typed_data();

#ifdef _OPENMP
  const int64_t max_threads = omp_get_max_threads();
#else
  const int64_t max_threads = 1;
#endif
  int n_threads = static_cast<int>(std::min(max_threads, batch_size));
#pragma omp parallel for num_threads(n_threads)
  for (int64_t b = 0; b < batch_size; ++b)
  {
    std::vector<int> row_mult(n_rows);
    std::transform(rows_ptr + b * n_rows, rows_ptr + (b + 1) * n_rows, row_mult.begin(),
                   [](uint64_t v) { return static_cast<int>(v); });
    std::vector<int> col_mult(n_cols);
    std::transform(cols_ptr + b * n_cols, cols_ptr + (b + 1) * n_cols, col_mult.begin(),
                   [](uint64_t v) { return static_cast<int>(v); });

    Matrix<std::complex<double>> matrix(n_rows, n_cols,
        A_ptr + b * n_rows * n_cols);

    std::complex<double> cot = cot_ptr[b];

    ComputePermBwd(matrix, row_mult, col_mult, cot,
                   ct_x_ptr + b * n_rows * n_cols);
  }

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(PermBwd, PermBwdImpl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::DataType::C128>>() // res
                                  .Arg<ffi::Buffer<ffi::DataType::C128>>() // A
                                  .Arg<ffi::Buffer<ffi::DataType::U64>>()  // rows
                                  .Arg<ffi::Buffer<ffi::DataType::U64>>()  // cols
                                  .Arg<ffi::Buffer<ffi::DataType::C128>>() // cotangent
                                  .Ret<ffi::Buffer<ffi::DataType::C128>>() // ct_x
);

template <typename T>
py::capsule EncapsulateFfiHandler(T *fn)
{
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be and XLA FFI handler");
  return py::capsule(reinterpret_cast<void *>(fn));
}

PYBIND11_MODULE(_perm_boost_core, m)
{
  m.doc() = "CPU permanent calculator (XLA FFI).";
  m.def("registrations", []()
        {
    py::dict registrations;
    registrations["perm"] = EncapsulateFfiHandler(Permanent);
    registrations["perm_fwd"] = EncapsulateFfiHandler(PermFwd);
    registrations["perm_bwd"] = EncapsulateFfiHandler(PermBwd);
    return registrations; });

  m.attr("__version__") = "dev";
}
