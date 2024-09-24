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

#ifndef LOOP_TORONTONIAN_H
#define LOOP_TORONTONIAN_H

#include "matrix.hpp"

/**
 * Calculates the torontonian of the input matrix.
 *
 * @param matrix_in The input matrix.
 * @returns The torontonian.
 */
template <typename TScalar>
extern TScalar loop_torontonian_cpp(
    Matrix<TScalar> &matrix,
    Vector<TScalar> &displacement_vector
);

#endif
