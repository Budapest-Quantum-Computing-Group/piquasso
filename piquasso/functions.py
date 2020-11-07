#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np


def gaussian_wigner_function(quadrature_matrix, *, d, mean, cov):
    assert len(quadrature_matrix[0]) == len(mean), (
        "'quadrature_matrix' elements should have the same dimension as 'mean': "
        f"dim(quadrature_matrix[0])={len(quadrature_matrix[0])}, dim(mean)={len(mean)}."
    )

    return [
        _gaussian_wigner_function_for_scalar(quadrature_array, d=d, mean=mean, cov=cov)
        for quadrature_array in quadrature_matrix
    ]


def _gaussian_wigner_function_for_scalar(X, *, d, mean, cov):
    return (
        (1 / (np.pi ** d))
        * np.sqrt((1 / np.linalg.det(cov)))
        * np.exp(
            - (X - mean)
            @ np.linalg.inv(cov)
            @ (X - mean)
        )
    ).real
