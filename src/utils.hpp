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

#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifndef HOST_DEVICE
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif
#endif

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

/**
 * @brief Computes a binomial coefficient without fixed-width integer overflow.
 *
 * Hafnian prefactors are accumulated in floating point, so calculating them in
 * an integer type first only reduces the supported occupation-number range.
 */
inline double binomialCoeffDouble(std::int64_t n, std::int64_t k)
{
    if (k < 0 || n < 0 || k > n)
        return 0.0;

    if (k > n - k)
        k = n - k;

    long double result = 1.0L;

    for (std::int64_t i = 1; i <= k; ++i)
    {
        result *= static_cast<long double>(n - k + i);
        result /= static_cast<long double>(i);
    }

    if (result > static_cast<long double>(std::numeric_limits<double>::max()))
        throw std::overflow_error("The binomial coefficient is too large.");

    return static_cast<double>(result);
}

/**
 * @brief Computes the binomial coefficient.
 *
 * This function calculates the binomial coefficient of n and k.
 *
 * @tparam int_type The type of the binomial coefficient.
 * @param n The first parameter of the binomial coefficient.
 * @param k The second parameter of the binomial coefficient.
 * @return The binomial coefficient of n and k.
 */
template <typename int_type>
int_type
binomialCoeffTemplated(int n, int k)
{
    std::vector<int_type> C(k + 1);
    C[0] = 1;
    for (int i = 1; i <= n; i++)
    {
        for (int j = std::min(i, k); j > 0; j--)
            C[j] = C[j] + C[j - 1];
    }
    return C[k];
}

/**
 * @brief Computes the binomial coefficient without using the standard library.
 *
 * This function calculates the binomial coefficient of n and k using manual loops.
 *
 * @tparam int_type The type of the binomial coefficient.
 * @param n The first parameter.
 * @param k The second parameter.
 * @return The binomial coefficient of n and k.
 */
template <typename int_type>
HOST_DEVICE int_type
binomialCoeffManual(int n, int k)
{
    if (k > n)
        return 0;
    int_type *C = new int_type[k + 1];

    for (int j = 0; j <= k; j++)
        C[j] = 0;
    C[0] = 1;

    for (int i = 1; i <= n; i++)
    {
        int end = (i < k) ? i : k;
        for (int j = end; j > 0; j--)
            C[j] = C[j] + C[j - 1];
    }
    int_type result = C[k];
    delete[] C;
    return result;
}

/**
 * @brief Computes the binomial coefficient.
 *
 * This function calculates the binomial coefficient of n and k.
 *
 * @param n The first parameter of the binomial coefficient.
 * @param k The second parameter of the binomial coefficient.
 * @return The binomial coefficient of n and k.
 */
inline int64_t binomialCoeffInt128(int n, int k)
{
    return binomialCoeffTemplated<int64_t>(n, k);
}

/**
 * @brief Computes the sum of a vector.
 *
 * This function calculates the sum of the elements in a vector.
 *
 * @tparam T The type of the elements in the vector.
 * @param vec The input vector.
 * @return The sum of the elements in the vector.
 */
template <typename T>
int sum(const std::vector<T> &vec)
{
    return std::accumulate(vec.begin(), vec.end(), 0);
}

#endif
