/*
 * Copyright 2021-2025 Budapest Quantum Computing Group
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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "numpy_utils.hpp"
#include "permanent.hpp"

namespace py = pybind11;

template <typename TScalar>
py::object permanent_np(
    py::array_t<std::complex<TScalar>, py::array::c_style> matrix,
    py::array_t<int, py::array::c_style | py::array::forcecast> row_mult_arr,
    py::array_t<int, py::array::c_style | py::array::forcecast> col_mult_arr
)
{
    Matrix<std::complex<TScalar>> native_matrix = numpy_to_matrix(matrix);

    Vector<int> row_mult = numpy_to_vector(row_mult_arr);
    Vector<int> col_mult = numpy_to_vector(col_mult_arr);

    std::complex<TScalar> result = permanent_cpp(native_matrix, row_mult, col_mult);

    return create_numpy_scalar(result);
}

const char* permanent_docstring = R""""(
Calculates the permanent of a matrix.

This function is based on the Parlett-Reid algorithm described in the following paper: https://arxiv.org/abs/1102.3440
    ...
)"""";

PYBIND11_MODULE(permanent, m)
{
    m.def("permanent", &permanent_np<float>, permanent_docstring,
        py::arg("matrix"), py::arg("rows"), py::arg("cols")
    );
    m.def("permanent", &permanent_np<double>, permanent_docstring,
        py::arg("matrix"), py::arg("rows"), py::arg("cols")
    );
}
