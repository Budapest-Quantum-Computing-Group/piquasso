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

#ifndef UTILS_H
#define UTILS_H

#include <numeric>
#include <vector>

/**
 * @brief Computes the binomial coefficient.
 *
 * This function calculates the binomial coefficient of n and k.
 *
 * @tparam TInt The type of the binomial coefficient.
 * @param n The first parameter of the binomial coefficient.
 * @param k The second parameter of the binomial coefficient.
 * @return The binomial coefficient of n and k.
 */
template <typename TInt>
TInt binomialCoeff(TInt n, TInt k)
{
    if (k < 0 || n < 0 || k > n)
        return TInt{0};

    if (k == 0 || k == n)
        return TInt{1};

    if (k > n - k)
        k = n - k;

    TInt result = 1;

    for (TInt i = 1; i <= k; ++i)
        result = (result / i) * (n - k + i) + (result % i) * (n - k + i) / i;

    return result;
}

#endif
