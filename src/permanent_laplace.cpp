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

#include "matrix.hpp"
#include "utils.hpp"
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

#include "n_aryGrayCodeCounter.hpp"
#include <thread>
#if defined(_OPENMP)
#include <omp.h>
#endif

/**
 * @brief Computes the permanents of the Laplace submatrices of a given matrix.
 *
 * The algorithm takes inspiration from the original implementation found under
 * `permanent.cpp`, but modified according to Lemma 1 from
 * [Faster classical Boson Sampling](https://arxiv.org/abs/2005.04214).
 *
 * @tparam TScalar The type of the elements in the matrix.
 * @param A The input matrix for which the permanent submatrices are to be computed.
 * @param rows A vector representing the multiplicity of each row.
 * @param cols A vector representing the multiplicity of each column (to be subtracted)
 * @return The permanents of the Laplace expansion submatrices (one for each column).
 */
template <typename T>
Vector<std::complex<T>> permanent_laplace_cpp(
    Matrix<std::complex<T>> &A, Vector<int> &rows, Vector<int> &cols
){
    using TComplex = std::complex<T>;

    int sum_rows = rows.sum();
    int sum_cols = cols.sum();

    if (A.rows == 0 || A.cols == 0 || sum_rows == 0 || sum_cols == 0)
    {
        Vector<TComplex> result(1);
        result[0] = TComplex(1.0, 0.0);
        return result;
    }

    size_t min_idx = 0;
    int minelem = 0;

    for (size_t i = 0; i < rows.size(); i++)
    {
        if (minelem == 0 || (rows[i] < minelem && rows[i] != 0))
        {
            minelem = rows[i];
            min_idx = i;
        }
    }

    if (rows.size() > 0 && minelem != 0)
    {
        Vector<int> rows_(rows.size() + 1);

        rows_[0] = 1;

        for (size_t i = 0; i < rows.size(); i++)
        {
            rows_[i + 1] = rows[i];
        }
        rows_[1 + min_idx] -= 1;

        Matrix<TComplex> mtx_(A.rows + 1, A.cols);

        for (size_t j = 0; j < A.cols; j++)
        {
            mtx_(0, j) = A(min_idx, j);
        }

        for (size_t i = 0; i < A.rows; i++)
        {
            for (size_t j = 0; j < A.cols; j++)
            {
                mtx_(i + 1, j) = A(i, j);
            }
        }

        rows = rows_;
        A = mtx_;
    }

    Matrix<TComplex> mtx2(A.rows, A.cols);
    for (size_t i = 0; i < A.size(); i++)
        mtx2[i] = A[i] * TComplex(2.0, 0.0);

    // Create a dynamically allocated array for n-ary limits.
    size_t n_ary_size = rows.size() - 1;
    int *n_ary_limits = new int[n_ary_size];
    for (size_t i = 0; i < n_ary_size; i++)
    {
        n_ary_limits[i] = rows[i + 1] + 1;
    }

    int64_t idx_max = n_ary_limits[0];
    for (size_t i = 1; i < n_ary_size; i++)
    {
        idx_max *= n_ary_limits[i];
    }

    // determine the concurrency of the calculation
    unsigned int n_threads = std::thread::hardware_concurrency();
    auto concurrency = static_cast<int64_t>(n_threads * 4);
    concurrency = concurrency < idx_max ? concurrency : idx_max;

    Matrix<TComplex> thread_results(static_cast<unsigned int>(concurrency), cols.size());

// See: https://stackoverflow.com/a/8447025
#if defined(_OPENMP)
#pragma omp parallel for num_threads(static_cast<unsigned int>(concurrency))
#endif
    for (int64_t job_idx = 0; job_idx < concurrency; job_idx++)
    {
        // initial offset and upper boundary of the gray code counter
        int64_t work_batch = idx_max / concurrency;
        int64_t initial_offset = job_idx * work_batch;
        int64_t offset_max = (job_idx + 1) * work_batch - 1;
        if (job_idx == concurrency - 1)
            offset_max = idx_max - 1;

        // Use the updated gray code counter, passing both the limits pointer and its size.
        n_aryGrayCodeCounter gcode_counter(n_ary_limits, n_ary_size, initial_offset);
        gcode_counter.set_offset_max(offset_max);

        // the gray code vector
        int *gcode = gcode_counter.get();

        // calculate the initial column sum and binomial coefficient
        int binomial_coeff = 1;

        Matrix<TComplex> colsum(1, cols.size());
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
                colsum[j] += static_cast<TComplex>(A(i + 1, j)) *
                             static_cast<T>(row_mult_current - 2 * minus_signs);

            minus_signs_all += minus_signs;

            // update the binomial coefficient
            binomial_coeff *= binomialCoeff<int>(row_mult_current, minus_signs);
        }

        // variable to refer to the parity of the delta vector (+1 if even, -1 if odd)
        char parity = (minus_signs_all % 2 == 0) ? 1 : -1;

        Matrix<TComplex> colsum_prods(1, cols.size());
        for (size_t l = 0; l < cols.size(); l++) {
            colsum_prods[l] = static_cast<T>(parity);
            for (size_t i = 0; i < cols.size(); i++)
                for (int j = 0; j < (i == l ? cols[i] - 1 : cols[i]); j++)
                    colsum_prods[l] *= colsum[i];
        }

        auto job_idx_uint = static_cast<unsigned int>(job_idx);

        // add the initial addend to the permanent - store in thread-local result
        for (size_t i = 0; i < cols.size(); i++)
            thread_results[job_idx_uint * cols.size() + i] = colsum_prods[i] * static_cast<T>(binomial_coeff);

        // iterate over gray codes to calculate permanent addends
        for (int64_t i = initial_offset + 1; i < offset_max + 1; i++)
        {
            int changed_index = 0;
            int prev_value = 0;
            int value = 0;
            if (gcode_counter.next(changed_index, prev_value, value))
                break;

            parity = -parity;

            // update column sum
            int row_offset = changed_index + 1;
            for (size_t j = 0; j < cols.size(); j++)
                colsum[j] += mtx2(row_offset, j) * static_cast<T>(prev_value - value);

            // calculate the product of the elements
            for (size_t l = 0; l < cols.size(); l++) {
                colsum_prods[l] = static_cast<T>(parity);
                for (size_t k = 0; k < cols.size(); k++)
                    for (int j = 0; j < (k == l ? cols[k] - 1 : cols[k]); j++)
                        colsum_prods[l] *= colsum[k];
            }


            int row_mult_current = rows[changed_index + 1];
            binomial_coeff =
                value < prev_value
                    ? binomial_coeff * prev_value / (row_mult_current - value)
                    : binomial_coeff * (row_mult_current - prev_value) / value;

            for (size_t k = 0; k < cols.size(); k++)
                thread_results[job_idx_uint * cols.size() + k] += colsum_prods[k] * static_cast<T>(binomial_coeff);
        }
    }

    Vector<TComplex> permanents(cols.size());
    auto denominator = static_cast<T>(ldexp(1.0, sum_rows - 1));

    for (size_t i = 0; i < cols.size(); i++) {
        permanents[i] = TComplex(0.0, 0.0);
        for (int64_t job_idx = 0; job_idx < concurrency; job_idx++)
            permanents[i] += thread_results[static_cast<size_t>(job_idx) * cols.size() + i];

        permanents[i] /= denominator;
    }

    delete[] n_ary_limits;

    return permanents;
}

template Vector<std::complex<float>> permanent_laplace_cpp<float>(
    Matrix<std::complex<float>> &matrix, Vector<int> &rows, Vector<int> &cols);
template Vector<std::complex<double>> permanent_laplace_cpp<double>(
    Matrix<std::complex<double>> &matrix, Vector<int> &rows, Vector<int> &cols);
