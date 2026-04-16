#ifndef UTILS_H
#define UTILS_H

#include <numeric>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

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
HOST_DEVICE
    int_type
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
int sum(std::vector<T> vec)
{
  return std::accumulate(vec.begin(), vec.end(), 0);
}

#endif