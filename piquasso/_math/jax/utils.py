#
# Copyright 2021-2024 Budapest Quantum Computing Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import jax
import jax.numpy as jnp


def factorial(n):
    return jnp.where(n < 0, 0, jax.lax.exp(jax.lax.lgamma(n + 1)))


# The following implementation of `eig` is copied with copyright from
# https://github.com/facebookresearch/fmmax/blob/8c7491a23e902644614c1761b6fbefa472786567/src/fmmax/utils.py#L101
# following discussions on https://github.com/jax-ml/jax/issues/2748.
# Minor modifications were applied.


# MIT License
#
# Copyright (c) 2023 Meta Research
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


EPS_EIG = 1e-6


@jax.custom_vjp
def eig(matrix: jnp.ndarray, eps: float = EPS_EIG) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Wraps `jnp.linalg.eig` in a jit-compatible, differentiable manner.

    The custom vjp allows gradients with resepct to the eigenvectors, unlike the
    standard jax implementation of `eig`. We use an expression for the gradient
    given in [2019 Boeddeker] along with a regularization scheme used in [2021
    Colburn]. The method effectively applies a Lorentzian broadening to a term
    containing the inverse difference of eigenvalues.

    [2019 Boeddeker] https://arxiv.org/abs/1701.00392
    [2021 Coluburn] https://www.nature.com/articles/s42005-021-00568-6

    Args:
        matrix: The matrix for which eigenvalues and eigenvectors are sought.
        eps: Parameter which determines the degree of broadening.

    Returns:
        The eigenvalues and eigenvectors.
    """
    del eps
    return _eig_host(matrix)


def _eig_host(matrix: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Wraps jnp.linalg.eig so that it can be jit-ed on a machine with GPUs."""
    eigenvalues_shape = jax.ShapeDtypeStruct(matrix.shape[:-1], matrix.dtype)
    eigenvectors_shape = jax.ShapeDtypeStruct(matrix.shape, matrix.dtype)

    def _eig_cpu(matrix: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # We force this computation to be performed on the cpu by jit-ing and
        # explicitly specifying the device.
        with jax.default_device(jax.devices("cpu")[0]):
            return jax.jit(jnp.linalg.eig)(matrix)

    return jax.pure_callback(
        _eig_cpu,
        (eigenvalues_shape, eigenvectors_shape),
        matrix,
    )


def _eig_fwd(
    matrix: jnp.ndarray,
    eps: float,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, float]]:
    """Implements the forward calculation for `eig`."""
    eigenvalues, eigenvectors = _eig_host(matrix)
    return (eigenvalues, eigenvectors), (eigenvalues, eigenvectors, eps)


def _eig_bwd(
    res: Tuple[jnp.ndarray, jnp.ndarray, float],
    grads: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, None]:
    """Implements the backward calculation for `eig`."""
    eigenvalues, eigenvectors, eps = res
    grad_eigenvalues, grad_eigenvectors = grads

    # Compute the F-matrix, from equation 5 of [2021 Colburn]. This applies a
    # Lorentzian broadening to the matrix `f = 1 / (eigenvalues[i] - eigenvalues[j])`.
    eigenvalues_i = eigenvalues[..., jnp.newaxis, :]
    eigenvalues_j = eigenvalues[..., :, jnp.newaxis]
    f_broadened = (eigenvalues_i - eigenvalues_j) / (
        (eigenvalues_i - eigenvalues_j) ** 2 + eps
    )

    # Manually set the diagonal elements to zero, as we do not use broadening here.
    i = jnp.arange(f_broadened.shape[-1])
    f_broadened = f_broadened.at[..., i, i].set(0)

    # By jax convention, gradients are with respect to the complex parameters, not with
    # respect to their conjugates. Take the conjugates.
    grad_eigenvalues_conj = jnp.conj(grad_eigenvalues)
    grad_eigenvectors_conj = jnp.conj(grad_eigenvectors)

    eigenvectors_H = jnp.conj(eigenvectors).T
    dim = eigenvalues.shape[-1]
    eye_mask = jnp.eye(dim, dtype=bool)
    eye_mask = eye_mask.reshape((1,) * (eigenvalues.ndim - 1) + (dim, dim))

    # Then, the gradient is found by equation 4.77 of [2019 Boeddeker].
    rhs = (
        jnp.diag(grad_eigenvalues_conj)
        + jnp.conj(f_broadened) * (eigenvectors_H @ grad_eigenvectors_conj)
        - jnp.conj(f_broadened)
        * (eigenvectors_H @ eigenvectors)
        @ jnp.where(eye_mask, jnp.real(eigenvectors_H @ grad_eigenvectors_conj), 0.0)
    ) @ eigenvectors_H
    grad_matrix = jnp.linalg.solve(eigenvectors_H, rhs)

    # Take the conjugate of the gradient, reverting to the jax convention
    # where gradients are with respect to complex parameters.
    grad_matrix = jnp.conj(grad_matrix)

    # Return `grad_matrix`, and `None` for the gradient with respect to `eps`.
    return grad_matrix, None


eig.defvjp(_eig_fwd, _eig_bwd)
