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
#include "pfaffian.hpp"

namespace py = pybind11;

template <typename TScalar>
py::object pfaffian_np(
    py::array_t<TScalar, py::array::c_style> matrix)
{
    Matrix<TScalar> native_matrix = numpy_to_matrix(matrix);

    TScalar result = pfaffian_cpp(native_matrix);

    return create_numpy_scalar(result);
}

const char* pfaffian_docstring = R""""(
Calculates the pfaffian of a matrix.

This function is based on the Parlett-Reid algorithm described in the following paper: https://arxiv.org/abs/1102.3440
    ...
)"""";

PYBIND11_MODULE(pfaffian, m)
{
    m.def("pfaffian", &pfaffian_np<float>, pfaffian_docstring);
    m.def("pfaffian", &pfaffian_np<double>, pfaffian_docstring);
}