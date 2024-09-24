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

#ifndef NUMPY_TO_MATRIX_H
#define NUMPY_TO_MATRIX_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "matrix.hpp"
#include "torontonian.hpp"

namespace py = pybind11;

/**
 * Create a numpy scalar from a C++ native data type.
 *
 * @note Not sure if this is the right way to create a numpy scalar. This just returns a
 * 0-dimensional array instead of a scalar, which is almost the same, but not quite.
 *
 * Source: https://stackoverflow.com/a/44682603
 */
template <typename TScalar>
py::object create_numpy_scalar(TScalar input)
{
    TScalar *ptr = new TScalar;
    (*ptr) = input;

    py::capsule free_when_done(ptr, [](void *f)
    {
        TScalar *ptr = reinterpret_cast<TScalar *>(f);
        delete ptr;
    });

    return py::array_t<TScalar>({}, {}, ptr, free_when_done);
}

/**
 * Creates a Matrix from a numpy array with shared memory.
 */
template <typename TScalar>
Matrix<TScalar> numpy_to_matrix(
    py::array_t<TScalar, py::array::c_style | py::array::forcecast> numpy_array)
{
    py::buffer_info bufferinfo = numpy_array.request();

    size_t rows = bufferinfo.shape[0];
    size_t cols = bufferinfo.shape[1];

    TScalar *data = static_cast<TScalar *>(bufferinfo.ptr);

    Matrix<TScalar> matrix(rows, cols, data);

    return matrix;
}


/**
 * Creates a Vector from a numpy array with shared memory.
 */
template <typename TScalar>
Vector<TScalar> numpy_to_vector(
    py::array_t<TScalar, py::array::c_style | py::array::forcecast> numpy_array)
{
    py::buffer_info bufferinfo = numpy_array.request();

    size_t size = bufferinfo.shape[0];

    TScalar *data = static_cast<TScalar *>(bufferinfo.ptr);

    Vector<TScalar> vector(size, data);

    return vector;
}

#endif
