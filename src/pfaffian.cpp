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

/**
 * @brief The Pfaffian implementation using the Parlett-Reid algorithm.
 *
 * See: https://arxiv.org/abs/1102.3440
 */

#include <cmath>

#include "matrix.hpp"
#include "pfaffian.hpp"

template <typename TScalar>
TScalar pfaffian_cpp(Matrix<TScalar> &matrix_in) {
    size_t n = matrix_in.cols;
    if(n == 0) return 1.0;
    if((n & 1) == 1) return 0;

    TScalar result = 1.0;

    size_t kp;
    for(size_t k = 0; k < (n - 1); k += 2) {
        kp = k + 1;

        for (size_t i = k + 2; i < n; i++) {
            size_t index_i = i * n + k;
            size_t index_kp = kp * n + k;
            
            if (std::abs(matrix_in[index_i]) > std::abs(matrix_in[index_kp])) {
                kp = i;
            }
        }

        if(kp != k + 1) {
            TScalar tmp;

            size_t k1_start = (k + 1) * n;
            size_t kp_start = kp * n;

            for(size_t i = 0; i < n; i++) 
            {
                size_t i_k1 = k1_start + i;
                size_t i_kp = kp_start + i;

                tmp = matrix_in[i_k1];
                matrix_in[i_k1] = matrix_in[i_kp];
                matrix_in[i_kp] = tmp;
            }
            
            for(size_t i = 0; i < n; i++) {
                size_t i_k1 = (i * n) + k + 1;
                size_t i_kp = (i * n) + kp;

                tmp = matrix_in[i_k1];
                matrix_in[i_k1] = matrix_in[i_kp];
                matrix_in[i_kp] = tmp;
            }

            result *= -1;
        }

        TScalar element = matrix_in[(k * n) + k + 1];

        if(element != 0) {
            result *= element;

            size_t tau_len = n - (k + 2);
            TScalar * tau = new TScalar[tau_len];

            for (size_t i = 0; i < tau_len; i++) {
                tau[i] = matrix_in[(k * n) + (k + 2 + i)] / element;
            }            
            
            if (k + 2 < n) {

                for(size_t i = k + 2; i < n; i++) {
                    for(size_t j = k + 2; j < n; j++) {
                        matrix_in[(i * n) + j] += 
                            (
                                (tau[i - (k + 2)] * matrix_in[(j * n) + k + 1]) - 
                                (tau[j - (k + 2)] * matrix_in[(i * n) + k + 1])
                            );
                    }
                }

            }
            
        } else {
            return 0.0;
        }

    }

    return result;
}

template float pfaffian_cpp<float>(Matrix<float> &matrix_in);
template double pfaffian_cpp<double>(Matrix<double> &matrix_in);