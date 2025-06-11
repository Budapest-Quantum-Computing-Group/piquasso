/*
 * Copyright 2021-2025 Budapest Quantum Computing Group
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

#ifndef PERMANENT_HPP
#define PERMANENT_HPP

#include "matrix.hpp"

/**
 * Calculates the permanent of the input matrix.
 *
 * @param matrix The input matrix.
 * @param rows
 * @param cols
 * @returns The permanent of the input matrix
 */
template <typename TScalar>
extern std::complex<TScalar> permanent_cpp(
    Matrix<std::complex<TScalar>> &matrix, Vector<int> &rows, Vector<int> &cols);

#endif
