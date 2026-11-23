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

#ifndef BS_OUTPUT_STATE_HPP
#define BS_OUTPUT_STATE_HPP

#include <complex>
#include <cstdint>
#include <vector>

#include "matrix.hpp"

/**
 * Calculates the state vector on the particle subspace defined by
 * `initial_state` (Algorithm 1, `SLOS_full`, from "Strong Simulation of Linear
 * Optical Processes"), pruned according to the postselection data.
 *
 * Occupations are indexed with `size_t` ranks, so the number of occupations of
 * `sum(initial_state)` particles in `d` modes must fit in `size_t`. This is not
 * checked.
 *
 * Instantiated for `std::complex<float>` and `std::complex<double>`.
 *
 * @param interferometer The `d x d` one-particle unitary.
 * @param initial_state Input occupation numbers of length `d`.
 * @param postselect_modes Postselected modes.
 * @param postselect_photons Occupation numbers of the postselected modes.
 */
template <typename TScalar>
Vector<TScalar> calculate_BS_output_state(
    const Matrix<TScalar> &interferometer,
    const std::vector<std::int32_t> &initial_state,
    const std::vector<std::int32_t> &postselect_modes,
    const std::vector<std::int32_t> &postselect_photons
);

#endif
