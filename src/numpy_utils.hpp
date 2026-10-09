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

#ifndef NUMPY_TO_MATRIX_H
#define NUMPY_TO_MATRIX_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <string>
#include <vector>

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
 * Create a numpy vector from a C++ native data type.
 *
 * Source: https://stackoverflow.com/a/44682603
 */
template <typename TScalar>
py::object create_numpy_vector(Vector<TScalar> input)
{
    auto size = input.size();
    TScalar *ptr = new TScalar[size];
    for (size_t i = 0; i < size; i++) {
        ptr[i] = input[i];
    }

    py::capsule free_when_done(ptr, [](void *f)
    {
        TScalar *ptr = reinterpret_cast<TScalar *>(f);
        delete[] ptr;
    });

    return py::array_t<TScalar>({size}, {sizeof(TScalar)}, ptr, free_when_done);
}

template <typename TScalar>
py::object create_numpy_vector(const std::vector<TScalar> &input)
{
    auto size = input.size();
    TScalar *ptr = new TScalar[size];
    for (size_t i = 0; i < size; i++) {
        ptr[i] = input[i];
    }

    py::capsule free_when_done(ptr, [](void *f)
    {
        TScalar *ptr = reinterpret_cast<TScalar *>(f);
        delete[] ptr;
    });

    return py::array_t<TScalar>({size}, {sizeof(TScalar)}, ptr, free_when_done);
}

/**
 * Creates a Matrix from a numpy array with shared memory.
 */
template <typename TScalar, int Flags>
Matrix<TScalar> numpy_to_matrix(
    py::array_t<TScalar, Flags> numpy_array)
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
template <typename TScalar, int Flags>
Vector<TScalar> numpy_to_vector(
    py::array_t<TScalar, Flags> numpy_array)
{
    py::buffer_info bufferinfo = numpy_array.request();

    size_t size = bufferinfo.shape[0];

    TScalar *data = static_cast<TScalar *>(bufferinfo.ptr);

    Vector<TScalar> vector(size, data);

    return vector;
}

/**
 * Creates a nonnegative Vector view over a one-dimensional NumPy array.
 * The input array must outlive the returned view.
 */
template <typename TScalar, int Flags>
Vector<TScalar> nonnegative_numpy_to_vector(
    py::array_t<TScalar, Flags> numpy_array,
    const char *argument_name
)
{
    if (numpy_array.ndim() != 1) {
        throw py::value_error(
            std::string(argument_name) + " must be one-dimensional"
        );
    }

    Vector<TScalar> result = numpy_to_vector(numpy_array);
    for (size_t index = 0; index < result.length; index++) {
        if (result[index] < 0) {
            throw py::value_error(
                std::string(argument_name) + " must be nonnegative"
            );
        }
    }

    return result;
}

/**
 * Converts a one-dimensional NumPy array to an owning nonnegative vector.
 */
template <typename TScalar>
std::vector<TScalar> nonnegative_vector_from_numpy(
    const py::array &numpy_array,
    const char *argument_name
)
{
    py::array_t<
        TScalar,
        py::array::c_style | py::array::forcecast
    > converted(numpy_array);
    const Vector<TScalar> input = nonnegative_numpy_to_vector(
        converted, argument_name
    );
    std::vector<TScalar> result(input.length);

    for (size_t index = 0; index < input.length; index++) {
        result[index] = input[index];
    }

    return result;
}

#endif
