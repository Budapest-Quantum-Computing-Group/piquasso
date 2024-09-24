/*
 * Copyright 2021-2024 Budapest Quantum Computing Group
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
#include "torontonian.hpp"
#include "loop_torontonian.hpp"

namespace py = pybind11;

template <typename TScalar>
py::object torontonian_np(
    py::array_t<TScalar, py::array::c_style | py::array::forcecast> matrix)
{
    Matrix<TScalar> native_matrix = numpy_to_matrix(matrix);

    TScalar result = torontonian_cpp(native_matrix);

    return create_numpy_scalar(result);
}

template <typename TScalar>
py::object loop_torontonian_np(
    py::array_t<TScalar, py::array::c_style | py::array::forcecast> matrix,
    py::array_t<TScalar, py::array::c_style | py::array::forcecast> displacement_vector)
{
    Matrix<TScalar> native_matrix = numpy_to_matrix(matrix);
    Vector<TScalar> native_displacement_vector = numpy_to_vector(displacement_vector);

    TScalar result = loop_torontonian_cpp(native_matrix, native_displacement_vector);

    return create_numpy_scalar(result);
}

const char* torontonian_docstring = R""""(
Calculates the torontonian of a matrix.

Note:
    This function expects arguments in a different ordering than usual. Usually, the
    inputs are in the xxpp-ordering, but for this implementation, one needs to provide
    data in the xpxp-ordering.
)"""";


const char* loop_torontonian_docstring = R""""(
Calculates the loop torontonian of a matrix and a displacement vector.

Note:
    This function expects arguments in a different ordering than usual. Usually, the
    inputs are in the xxpp-ordering, but for this implementation, one needs to provide
    data in the xpxp-ordering.
)"""";

PYBIND11_MODULE(torontonian, m)
{
    m.def("torontonian", &torontonian_np<float>, torontonian_docstring);
    m.def("torontonian", &torontonian_np<double>, torontonian_docstring);
    m.def("loop_torontonian", &loop_torontonian_np<float>, loop_torontonian_docstring);
    m.def("loop_torontonian", &loop_torontonian_np<double>, loop_torontonian_docstring);
}
