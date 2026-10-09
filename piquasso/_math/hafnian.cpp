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

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "hafnian/loop_hafnian.hpp"
#include "hafnian/plain_hafnian.hpp"
#include "matrix.hpp"
#include "numpy_utils.hpp"
#include "validations.hpp"

namespace py = pybind11;

namespace {

using piquasso::hafnian::hafnian;
using piquasso::hafnian::hafnian_batch;
using piquasso::hafnian::loop_hafnian;
using piquasso::hafnian::loop_hafnian_batch;

void validate_common_inputs(
    const py::array &matrix,
    const py::array &occupations,
    std::int64_t cutoff = -1
) {
    if (matrix.ndim() != 2 || matrix.shape(0) != matrix.shape(1)) {
        throw py::value_error("matrix must be a square two-dimensional array");
    }
    if (occupations.ndim() != 1 || occupations.shape(0) != matrix.shape(0)) {
        throw py::value_error(
            "occupation_numbers must be one-dimensional and match matrix"
        );
    }
    if (cutoff == 0) {
        throw py::value_error("cutoff must be positive");
    }
}

py::object hafnian_numpy(
    const py::array &matrix,
    const py::array &occupation_numbers
) {
    validate_common_inputs(matrix, occupation_numbers);
    const std::vector<std::int64_t> occupations =
        nonnegative_vector_from_numpy<std::int64_t>(
            occupation_numbers, "occupation_numbers"
        );

    if (is_complex_array(matrix)) {
        py::array_t<
            std::complex<double>,
            py::array::c_style | py::array::forcecast
        > contiguous_matrix(matrix);
        const Matrix<std::complex<double>> native_matrix =
            numpy_to_matrix(contiguous_matrix);
        std::complex<double> result;
        {
            py::gil_scoped_release release;
            result = hafnian(native_matrix, occupations);
        }
        return py::cast(result);
    }

    py::array_t<double, py::array::c_style | py::array::forcecast>
        contiguous_matrix(matrix);
    const Matrix<double> native_matrix = numpy_to_matrix(contiguous_matrix);
    double result;
    {
        py::gil_scoped_release release;
        result = hafnian(native_matrix, occupations);
    }
    return py::cast(result);
}

py::object loop_hafnian_numpy(
    const py::array &matrix,
    const py::array &diagonal,
    const py::array &occupation_numbers
) {
    validate_common_inputs(matrix, occupation_numbers);
    if (diagonal.ndim() != 1 || diagonal.shape(0) != matrix.shape(0)) {
        throw py::value_error("diagonal must be one-dimensional and match matrix");
    }
    const std::vector<std::int64_t> occupations =
        nonnegative_vector_from_numpy<std::int64_t>(
            occupation_numbers, "occupation_numbers"
        );

    if (is_complex_array(matrix) || is_complex_array(diagonal)) {
        py::array_t<
            std::complex<double>,
            py::array::c_style | py::array::forcecast
        > contiguous_matrix(matrix);
        const Matrix<std::complex<double>> native_matrix =
            numpy_to_matrix(contiguous_matrix);
        py::array_t<
            std::complex<double>,
            py::array::c_style | py::array::forcecast
        > contiguous_diagonal(diagonal);
        const Vector<std::complex<double>> native_diagonal =
            numpy_to_vector(contiguous_diagonal);
        std::complex<double> result;
        {
            py::gil_scoped_release release;
            result = loop_hafnian(native_matrix, native_diagonal, occupations);
        }
        return py::cast(result);
    }

    py::array_t<double, py::array::c_style | py::array::forcecast>
        contiguous_matrix(matrix);
    const Matrix<double> native_matrix = numpy_to_matrix(contiguous_matrix);
    py::array_t<double, py::array::c_style | py::array::forcecast>
        contiguous_diagonal(diagonal);
    const Vector<double> native_diagonal = numpy_to_vector(contiguous_diagonal);
    double result;
    {
        py::gil_scoped_release release;
        result = loop_hafnian(native_matrix, native_diagonal, occupations);
    }
    return py::cast(result);
}

py::object hafnian_batch_numpy(
    const py::array &matrix,
    const py::array &occupation_numbers,
    std::int64_t cutoff
) {
    validate_common_inputs(matrix, occupation_numbers, cutoff);
    if (cutoff < 1) {
        throw py::value_error("cutoff must be positive");
    }
    const std::vector<std::int64_t> occupations =
        nonnegative_vector_from_numpy<std::int64_t>(
            occupation_numbers, "occupation_numbers"
        );

    if (is_complex_array(matrix)) {
        py::array_t<
            std::complex<double>,
            py::array::c_style | py::array::forcecast
        > contiguous_matrix(matrix);
        const Matrix<std::complex<double>> native_matrix =
            numpy_to_matrix(contiguous_matrix);
        std::vector<std::complex<double>> result;
        {
            py::gil_scoped_release release;
            result = hafnian_batch(native_matrix, occupations, cutoff);
        }
        return create_numpy_vector(result);
    }

    py::array_t<double, py::array::c_style | py::array::forcecast>
        contiguous_matrix(matrix);
    const Matrix<double> native_matrix = numpy_to_matrix(contiguous_matrix);
    std::vector<double> result;
    {
        py::gil_scoped_release release;
        result = hafnian_batch(native_matrix, occupations, cutoff);
    }
    return create_numpy_vector(result);
}

py::object loop_hafnian_batch_numpy(
    const py::array &matrix,
    const py::array &diagonal,
    const py::array &occupation_numbers,
    std::int64_t cutoff
) {
    validate_common_inputs(matrix, occupation_numbers, cutoff);
    if (diagonal.ndim() != 1 || diagonal.shape(0) != matrix.shape(0)) {
        throw py::value_error("diagonal must be one-dimensional and match matrix");
    }
    if (cutoff < 1) {
        throw py::value_error("cutoff must be positive");
    }
    const std::vector<std::int64_t> occupations =
        nonnegative_vector_from_numpy<std::int64_t>(
            occupation_numbers, "occupation_numbers"
        );

    if (is_complex_array(matrix) || is_complex_array(diagonal)) {
        py::array_t<
            std::complex<double>,
            py::array::c_style | py::array::forcecast
        > contiguous_matrix(matrix);
        const Matrix<std::complex<double>> native_matrix =
            numpy_to_matrix(contiguous_matrix);
        py::array_t<
            std::complex<double>,
            py::array::c_style | py::array::forcecast
        > contiguous_diagonal(diagonal);
        const Vector<std::complex<double>> native_diagonal =
            numpy_to_vector(contiguous_diagonal);
        std::vector<std::complex<double>> result;
        {
            py::gil_scoped_release release;
            result = loop_hafnian_batch(
                native_matrix, native_diagonal, occupations, cutoff
            );
        }
        return create_numpy_vector(result);
    }

    py::array_t<double, py::array::c_style | py::array::forcecast>
        contiguous_matrix(matrix);
    const Matrix<double> native_matrix = numpy_to_matrix(contiguous_matrix);
    py::array_t<double, py::array::c_style | py::array::forcecast>
        contiguous_diagonal(diagonal);
    const Vector<double> native_diagonal = numpy_to_vector(contiguous_diagonal);
    std::vector<double> result;
    {
        py::gil_scoped_release release;
        result = loop_hafnian_batch(
            native_matrix, native_diagonal, occupations, cutoff
        );
    }
    return create_numpy_vector(result);
}

} // namespace

PYBIND11_MODULE(hafnian, module) {
    module.doc() = R"doc(Native hafnian and loop-hafnian implementations.

Most of the code in this package consists of translated versions of the PiquassoBoost
C++ code: https://github.com/Budapest-Quantum-Computing-Group/piquassoboost.

The algorithms are further enhanced to account for repetitions, as described in
https://arxiv.org/abs/2108.01622.
)doc";

    module.def(
        "hafnian_with_reduction",
        &hafnian_numpy,
        py::arg("matrix"),
        py::arg("occupation_numbers"),
        "Calculate a hafnian of a symmetric matrix while factoring in "
        "occupation-number repetitions."
    );
    module.def(
        "hafnian_with_reduction_batch",
        &hafnian_batch_numpy,
        py::arg("matrix"),
        py::arg("occupation_numbers"),
        py::arg("cutoff"),
        "Calculate a batch of hafnians of a symmetric matrix in the final mode."
    );
    module.def(
        "loop_hafnian_with_reduction",
        &loop_hafnian_numpy,
        py::arg("matrix"),
        py::arg("diagonal"),
        py::arg("occupation_numbers"),
        "Calculate a loop hafnian of a symmetric matrix with "
        "occupation-number repetitions."
    );
    module.def(
        "loop_hafnian_with_reduction_batch",
        &loop_hafnian_batch_numpy,
        py::arg("matrix"),
        py::arg("diagonal"),
        py::arg("occupation_numbers"),
        py::arg("cutoff"),
        "Calculate a batch of loop hafnians of a symmetric matrix in the "
        "final mode."
    );
}
