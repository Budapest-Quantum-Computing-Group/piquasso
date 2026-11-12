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

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <vector>

#include "hafnian.hpp"
#include "matrix.hpp"
#include "numpy_utils.hpp"
#include "validations.hpp"

namespace py = pybind11;

namespace {

using piquasso::hafnian::loop_hafnian;
using piquasso::hafnian::loop_hafnian_batch;

using Complex = std::complex<double>;
using ContiguousArray =
    py::array_t<Complex, py::array::c_style | py::array::forcecast>;

void validate_inputs(
    const py::array &matrix,
    const py::array &diagonal,
    const py::array &occupation_numbers
) {
    validate_square_matrix(matrix);
    validate_vector_matching_matrix(
        occupation_numbers, matrix, "occupation_numbers"
    );
    validate_vector_matching_matrix(diagonal, matrix, "diagonal");
}

bool is_single_precision(const py::array &array) {
    return py::isinstance<py::array_t<float>>(array)
        || py::isinstance<py::array_t<std::complex<float>>>(array);
}

// The results are calculated in double-precision complex arithmetic. They are
// returned as real numbers for real inputs, and in single precision if all
// inputs are in single precision.
bool has_real_result(const py::array &matrix, const py::array &diagonal) {
    return !is_complex_array(matrix) && !is_complex_array(diagonal);
}

bool has_single_precision_result(
    const py::array &matrix,
    const py::array &diagonal
) {
    return is_single_precision(matrix) && is_single_precision(diagonal);
}

py::object to_input_type(
    const Complex &value,
    const py::array &matrix,
    const py::array &diagonal
) {
    const bool single_precision = has_single_precision_result(matrix, diagonal);
    if (has_real_result(matrix, diagonal)) {
        return single_precision ? py::dtype::of<float>().attr("type")(value.real())
                                : py::cast(value.real());
    }
    return single_precision
        ? py::dtype::of<std::complex<float>>().attr("type")(value)
        : py::cast(value);
}

py::object to_input_type(
    const std::vector<Complex> &values,
    const py::array &matrix,
    const py::array &diagonal
) {
    const bool real = has_real_result(matrix, diagonal);
    py::object array;
    if (real) {
        std::vector<double> real_parts(values.size());
        std::transform(
            values.begin(), values.end(), real_parts.begin(),
            [](const Complex &value) { return value.real(); }
        );
        array = create_numpy_vector(real_parts);
    } else {
        array = create_numpy_vector(values);
    }
    if (has_single_precision_result(matrix, diagonal)) {
        return array.attr("astype")(
            real ? py::dtype::of<float>() : py::dtype::of<std::complex<float>>()
        );
    }
    return array;
}

py::object loop_hafnian_numpy(
    const py::array &matrix,
    const py::array &diagonal,
    const py::array &occupation_numbers
) {
    validate_inputs(matrix, diagonal, occupation_numbers);
    const std::vector<std::int64_t> occupations =
        nonnegative_vector_from_numpy<std::int64_t>(
            occupation_numbers, "occupation_numbers"
        );
    const ContiguousArray contiguous_matrix(matrix);
    const ContiguousArray contiguous_diagonal(diagonal);
    const Matrix<Complex> native_matrix = numpy_to_matrix(contiguous_matrix);
    const Vector<Complex> native_diagonal = numpy_to_vector(contiguous_diagonal);

    Complex result;
    {
        py::gil_scoped_release release;
        result = loop_hafnian(native_matrix, native_diagonal, occupations);
    }

    return to_input_type(result, matrix, diagonal);
}

py::object loop_hafnian_batch_numpy(
    const py::array &matrix,
    const py::array &diagonal,
    const py::array &occupation_numbers,
    std::int64_t cutoff
) {
    validate_inputs(matrix, diagonal, occupation_numbers);
    validate_positive(cutoff, "cutoff");
    const std::vector<std::int64_t> occupations =
        nonnegative_vector_from_numpy<std::int64_t>(
            occupation_numbers, "occupation_numbers"
        );
    const ContiguousArray contiguous_matrix(matrix);
    const ContiguousArray contiguous_diagonal(diagonal);
    const Matrix<Complex> native_matrix = numpy_to_matrix(contiguous_matrix);
    const Vector<Complex> native_diagonal = numpy_to_vector(contiguous_diagonal);

    std::vector<Complex> result;
    {
        py::gil_scoped_release release;
        result = loop_hafnian_batch(
            native_matrix, native_diagonal, occupations, cutoff
        );
    }

    return to_input_type(result, matrix, diagonal);
}

// The hafnian is the loop hafnian with zero diagonal, of the precision of the
// matrix.
py::array zero_diagonal(const py::array &matrix) {
    validate_square_matrix(matrix);
    py::array diagonal(
        is_single_precision(matrix) ? py::dtype::of<float>()
                                    : py::dtype::of<double>(),
        py::array::ShapeContainer{matrix.shape(0)}
    );
    std::memset(
        diagonal.mutable_data(), 0, static_cast<std::size_t>(diagonal.nbytes())
    );
    return diagonal;
}

py::object hafnian_numpy(
    const py::array &matrix,
    const py::array &occupation_numbers
) {
    return loop_hafnian_numpy(matrix, zero_diagonal(matrix), occupation_numbers);
}

py::object hafnian_batch_numpy(
    const py::array &matrix,
    const py::array &occupation_numbers,
    std::int64_t cutoff
) {
    return loop_hafnian_batch_numpy(
        matrix, zero_diagonal(matrix), occupation_numbers, cutoff
    );
}

} // namespace

PYBIND11_MODULE(hafnian, module) {
    module.doc() = R"doc(Native hafnian and loop-hafnian implementations.

The loop hafnians are calculated by a parallel finite-difference Schur recursion in
O(n^2 2^n) time for 2n vertices, with repeated rows grouped.

The idea behind this algorithm will be described in a forthcoming arXiv preprint.
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
