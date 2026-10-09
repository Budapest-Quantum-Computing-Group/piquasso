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

#ifndef PIQUASSO_SRC_HAFNIAN_LABUDDE_HPP
#define PIQUASSO_SRC_HAFNIAN_LABUDDE_HPP

#include <cstddef>
#include <vector>

#include "matrix.hpp"

namespace piquasso::hafnian {

template <typename T>
void labudde(
    const Matrix<T> &matrix,
    Matrix<T> &coefficients,
    std::vector<T> &beta_products
) {
    const std::size_t dimension = matrix.rows;
    coefficients.rows = dimension;
    coefficients.cols = dimension;
    beta_products.resize(dimension);

    if (dimension == 0) {
        return;
    }

    coefficients(0, 0) = -matrix(0, 0);
    if (dimension == 1) {
        return;
    }

    coefficients(1, 0) = coefficients(0, 0) - matrix(1, 1);
    coefficients(1, 1) =
        matrix(0, 0) * matrix(1, 1) - matrix(0, 1) * matrix(1, 0);

    for (std::size_t index = 2; index < dimension; ++index) {
        beta_products[0] = matrix(index, index - 1);
        for (std::size_t product_index = 1;
             product_index < index;
             ++product_index) {
            beta_products[product_index] = beta_products[product_index - 1]
                * matrix(index - product_index, index - product_index - 1);
        }

        coefficients(index, 0) =
            coefficients(index - 1, 0) - matrix(index, index);

        for (std::size_t coefficient_index = 1;
             coefficient_index < index;
             ++coefficient_index) {
            T correction{};
            for (std::size_t offset = 1;
                 offset < coefficient_index;
                 ++offset) {
                correction += matrix(index - offset, index)
                    * beta_products[offset - 1]
                    * coefficients(
                        index - offset - 1,
                        coefficient_index - offset - 1
                    );
            }
            correction += matrix(index - coefficient_index, index)
                * beta_products[coefficient_index - 1];

            coefficients(index, coefficient_index) =
                coefficients(index - 1, coefficient_index)
                - matrix(index, index)
                    * coefficients(index - 1, coefficient_index - 1)
                - correction;
        }

        T correction{};
        for (std::size_t offset = 1; offset < index; ++offset) {
            correction += matrix(index - offset, index)
                * beta_products[offset - 1]
                * coefficients(index - offset - 1, index - offset - 1);
        }

        coefficients(index, index) =
            -matrix(index, index) * coefficients(index - 1, index - 1)
            - correction - matrix(0, index) * beta_products[index - 1];
    }
}

} // namespace piquasso::hafnian

#endif
