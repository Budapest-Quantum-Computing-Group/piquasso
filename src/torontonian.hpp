#ifndef TORONTONIAN_H
#define TORONTONIAN_H

#include "matrix.hpp"

/**
 * Calculates the torontonian of the input matrix.
 *
 * @param matrix_in The input matrix.
 * @returns The torontonian.
 */
template <typename TScalar>
extern TScalar torontonian_cpp(Matrix<TScalar> &matrix);

#endif