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

#ifndef PIQUASSO_NEUMAIER_HPP
#define PIQUASSO_NEUMAIER_HPP

#include <cmath>
#include <complex>

// Neumaier's compensated summation, which keeps the rounding error of every
// addition in a separate correction term.
//
// TODO: Use it in the permanent and the torontonian as well.

/**
 * @brief The state of Neumaier's summation, see `compensated_add`.
 *
 * @tparam T The type of the summands.
 */
template <typename T>
struct NeumaierSum {
    T sum{};
    T correction{};
};

/**
 * @brief The state of Neumaier's summation of complex numbers, with the real
 * and imaginary parts compensated separately.
 */
template <typename T>
struct NeumaierSum<std::complex<T>> {
    NeumaierSum<T> real;
    NeumaierSum<T> imag;
};

/**
 * @brief Adds a value to a Neumaier sum.
 */
template <typename T>
void compensated_add(NeumaierSum<T> &state, T value) {
    const T total = state.sum + value;
    if (std::abs(state.sum) >= std::abs(value))
        state.correction += (state.sum - total) + value;
    else
        state.correction += (value - total) + state.sum;

    state.sum = total;
}

template <typename T>
void compensated_add(
    NeumaierSum<std::complex<T>> &state,
    const std::complex<T> &value
) {
    compensated_add(state.real, value.real());
    compensated_add(state.imag, value.imag());
}

/**
 * @brief The value of a Neumaier sum.
 */
template <typename T>
T compensated_value(const NeumaierSum<T> &state) {
    return state.sum + state.correction;
}

template <typename T>
std::complex<T> compensated_value(const NeumaierSum<std::complex<T>> &state) {
    return {compensated_value(state.real), compensated_value(state.imag)};
}

#endif
