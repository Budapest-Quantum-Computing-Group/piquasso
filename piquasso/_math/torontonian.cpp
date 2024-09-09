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

namespace py = pybind11;

template <typename TScalar>
py::object torontonian_np(py::array_t<TScalar, py::array::c_style | py::array::forcecast> array)
{
    Matrix<TScalar> matrix = numpy_to_matrix(array);

    TScalar result = torontonian_cpp(matrix);

    return create_numpy_scalar(result);
}

PYBIND11_MODULE(torontonian, m)
{
    m.def("torontonian", &torontonian_np<float>);
    m.def("torontonian", &torontonian_np<double>);
}
