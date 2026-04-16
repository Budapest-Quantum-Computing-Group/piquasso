#ifndef PERMANENT_HPP
#define PERMANENT_HPP

#include "../matrix.hpp"
#include "../utils.hpp"
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

#include "../n_aryGrayCodeCounter.hpp"
#include <thread>
#include <omp.h>

#ifdef __SIZEOF_INT128__ // Check if __int128 is supported
using int_type = __int128;
#else
using int_type = int64_t; // Fallback to int64_t
#endif

/**
 * @brief Computes the permanent of a given matrix.
 *
 * This function calculates the permanent of the input matrix. The permanent is
 * a function similar to the determinant but with all signs positive in the
 * expansion by minors.
 *
 * @tparam T The type of the elements in the matrix.
 * @param mtx The input matrix for which the permanent is to be computed.
 * @param rows A vector representing the multiplicity of each row.
 * @param cols A vector representing the multiplicity of each column.
 * @return The permanent of the input matrix.
 */
template <typename T, typename precision_type>
std::complex<double> permanent(Matrix<std::complex<double>> &A,
                               std::vector<int> &rows, std::vector<int> &cols)
{

  size_t min_idx = 0;
  int minelem = 0;

  for (int i = 0; i < rows.size(); i++)
  {
    if (minelem == 0 || (rows[i] < minelem && rows[i] != 0))
    {
      minelem = rows[i];
      min_idx = i;
    }
  }

  if (rows.size() > 0 && minelem != 0)
  {
    std::vector<int> rows_(rows.size() + 1);

    rows_[0] = 1;

    for (int i = 0; i < rows.size(); i++)
    {
      rows_[i + 1] = rows[i];
    }
    rows_[1 + min_idx] -= 1;

    Matrix<T> mtx_(A.rows + 1, A.cols);

    for (int j = 0; j < A.cols; j++)
    {
      mtx_(0, j) = A(min_idx, j);
    }

    for (int i = 0; i < A.rows; i++)
    {
      for (int j = 0; j < A.cols; j++)
      {
        mtx_(i + 1, j) = A(i, j);
      }
    }

    rows = rows_;
    A = mtx_;
  }

  int sum_rows = sum(rows);
  int sum_cols = sum(cols);

  if (sum_rows != sum_cols)
  {
    std::string error("Number of input and output states should be equal");
    throw error;
  }

  if (A.rows == 0 || A.cols == 0 || sum_rows == 0 || sum_cols == 0)
    // the permanent of an empty matrix is 1 by definition
    return std::complex<double>(1.0, 0.0);

  if (A.rows == 1)
  {
    T ret(1.0, 0.0);
    for (size_t i = 0; i < cols.size(); i++)
    {
      for (size_t j = 0; j < cols[i]; j++)
      {
        ret *= A[i];
      }
    }

    return std::complex<double>(ret.real(), ret.imag());
  }

  Matrix<std::complex<double>> mtx2(A.rows, A.cols);
  for (size_t i = 0; i < A.size(); i++)
  {
    mtx2[i] = A[i] * 2.0;
  }

  // Create a dynamically allocated array for n-ary limits.
  size_t n_ary_size = rows.size() - 1;
  int *n_ary_limits = new int[n_ary_size];
  for (size_t i = 0; i < n_ary_size; i++)
  {
    n_ary_limits[i] = rows[i + 1] + 1;
  }

  uint64_t idx_max = n_ary_limits[0];
  for (size_t i = 1; i < n_ary_size; i++)
  {
    idx_max *= n_ary_limits[i];
  }

  const uint64_t MAX_IDX_MAX = 100000000;
  if (idx_max > MAX_IDX_MAX)
  {
    throw std::runtime_error("Problem too large: idx_max exceeds safe limit.");
  }

  // determine the concurrency of the calculation
  unsigned int n_threads = std::thread::hardware_concurrency();
  int64_t concurrency = static_cast<int64_t>(n_threads) * 4;
  concurrency = concurrency < idx_max ? concurrency : static_cast<int64_t>(idx_max);

  std::vector<T> thread_results(concurrency, T(0.0, 0.0));

#pragma omp parallel for num_threads(concurrency)
  for (int64_t job_idx = 0; job_idx < concurrency; job_idx++)
  {
    // initial offset and upper boundary of the gray code counter
    int64_t work_batch = idx_max / concurrency;
    int64_t initial_offset = job_idx * work_batch;
    int64_t offset_max = (job_idx + 1) * work_batch - 1;
    if (job_idx == concurrency - 1)
    {
      offset_max = idx_max - 1;
    }

    // Use the updated gray code counter, passing both the limits pointer and its size.
    n_aryGrayCodeCounter gcode_counter(n_ary_limits, n_ary_size, initial_offset);
    gcode_counter.set_offset_max(offset_max);

    // the gray code vector
    int *gcode = gcode_counter.get();

    // calculate the initial column sum and binomial coefficient
    int_type binomial_coeff = 1;

    Matrix<T> colsum(1, cols.size());
    std::uninitialized_copy_n(A.data, colsum.size(), colsum.data);

    // variable to count all the -1 elements in the delta vector
    int minus_signs_all = 0;

    for (size_t i = 0; i < n_ary_size; i++)
    {
      // the value of the element of the gray code stands for the number of
      // \delta_i=-1 elements in the subset of multiplicated rows
      const int &minus_signs = gcode[i];
      int row_mult_current = rows[i + 1];

      for (size_t j = 0; j < cols.size(); j++)
      {
        colsum[j] += static_cast<T>(A(i + 1, j)) *
                     static_cast<precision_type>(row_mult_current - 2 * minus_signs);
      }

      minus_signs_all += minus_signs;

      // update the binomial coefficient
      binomial_coeff *= binomialCoeffInt128(row_mult_current, minus_signs);
    }

    // variable to refer to the parity of the delta vector (+1 if even, -1 if odd)
    char parity = (minus_signs_all % 2 == 0) ? 1 : -1;

    T colsum_prod(static_cast<precision_type>(parity), static_cast<precision_type>(0.0));
    for (size_t i = 0; i < cols.size(); i++)
    {
      for (size_t j = 0; j < cols[i]; j++)
      {
        colsum_prod *= colsum[i];
      }
    }

    // add the initial addend to the permanent - store in thread-local result
    T &addend_loc = thread_results[job_idx];
    addend_loc += colsum_prod * static_cast<precision_type>(binomial_coeff);

    // iterate over gray codes to calculate permanent addends
    for (int64_t i = initial_offset + 1; i < offset_max + 1; i++)
    {
      int changed_index, prev_value, value;
      if (gcode_counter.next(changed_index, prev_value, value))
      {
        break;
      }

      parity = -parity;

      // update column sum and calculate the product of the elements
      int row_offset = changed_index + 1;
      T colsum_prod(static_cast<precision_type>(parity), static_cast<precision_type>(0.0));
      for (size_t j = 0; j < cols.size(); j++)
      {
        colsum[j] += mtx2(row_offset, j) *
                     static_cast<precision_type>(prev_value - value);
        for (size_t k = 0; k < cols[j]; k++)
        {
          colsum_prod *= colsum[j];
        }
      }

      int row_mult_current = rows[changed_index + 1];
      binomial_coeff =
          value < prev_value
              ? binomial_coeff * prev_value / (row_mult_current - value)
              : binomial_coeff * (row_mult_current - prev_value) / value;

      addend_loc += colsum_prod * static_cast<precision_type>(binomial_coeff);
    }
  }

  T permanent(0.0, 0.0);
  for (const auto &result : thread_results)
  {
    permanent += result;
  }

  permanent /= static_cast<precision_type>(ldexp(1.0, sum(rows) - 1));

  delete[] n_ary_limits;

  return permanent;
}

Matrix<std::complex<double>> grad_perm(Matrix<std::complex<double>> &A,
                                       std::vector<int> &rows,
                                       std::vector<int> &cols)
{
  int n = rows.size();

  Matrix<std::complex<double>> perm_grad(n, n);

#pragma omp parallel for collapse(2)
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      if (rows[i] == 0 || cols[j] == 0)
        continue;

      std::vector<int> grad_rows(rows);
      grad_rows[i] -= 1;

      Matrix<std::complex<double>> A_(A);

      std::vector<int> grad_cols(cols);
      grad_cols[j] -= 1;

      perm_grad(i, j) =
          static_cast<double>(rows[i]) * static_cast<double>(cols[j]) *
          permanent<std::complex<double>, double>(A_, grad_rows, grad_cols);
    }
  }

  return perm_grad;
}

#endif