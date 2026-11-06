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

#ifndef PIQUASSO_VALIDATIONS_HPP
#define PIQUASSO_VALIDATIONS_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <string>

inline bool is_complex_array(const pybind11::array &array) {
    return pybind11::str(array.dtype().attr("kind")).cast<std::string>() == "c";
}

inline void validate_square_matrix(
    const pybind11::array &matrix,
    const std::string &name = "matrix"
) {
    if (matrix.ndim() != 2 || matrix.shape(0) != matrix.shape(1)) {
        throw pybind11::value_error(
            name + " must be a square two-dimensional array"
        );
    }
}

inline void validate_vector_matching_matrix(
    const pybind11::array &vector,
    const pybind11::array &matrix,
    const std::string &name
) {
    if (vector.ndim() != 1 || vector.shape(0) != matrix.shape(0)) {
        throw pybind11::value_error(
            name + " must be one-dimensional and match matrix"
        );
    }
}

inline void validate_positive(std::int64_t value, const std::string &name) {
    if (value < 1) {
        throw pybind11::value_error(name + " must be positive");
    }
}

#endif
