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

import numpy as np

from functools import partial

from ..calculator import BuiltinCalculator


class JaxCalculator(BuiltinCalculator):
    """The calculations for a simulation using JAX.

    Example usage::

        import numpy as np
        import piquasso as pq
        from jax import jit, grad

        def example_func(r, theta):
            jax_calculator = pq.JaxCalculator()

            simulator = pq.PureFockSimulator(
                d=2,
                config=pq.Config(cutoff=5, dtype=np.float32, normalize=False),
                calculator=jax_calculator,
            )

            with pq.Program() as program:
                pq.Q() | pq.Vacuum()

                pq.Q(0) | pq.Displacement(r=r)

                pq.Q(0, 1) | pq.Beamsplitter(theta)

            state = simulator.execute(program).state

            return state.fock_probabilities[0]

        compiled_func = jit(example_func)

        vacuum_probability = compiled_func(0.1, np.pi / 7)

        compiled_grad_func = jit(grad(compiled_func))

        vacuum_probability_grad = compiled_grad_func(0.2, np.pi / 11)

    Note:
        This feature is still experimental.

    Note:
        Currently JIT compilation only works with the config variable `normalize=False`.

    Note:
        Only CPU calculations are supported currently.
    """

    def __init__(self):
        try:
            import jax.numpy as jnp
            import jax
        except ImportError:
            raise ImportError(
                "You have invoked a feature which requires 'jax'.\n"
                "You can install JAX via:\n"
                "\n"
                "pip install piquasso[jax]"
            )

        # NOTE: Because lots of calculations are still done using NumPy, and NumPy
        # prefers double precision, Piquasso uses double precision too, therefore it is
        # better to enable this config variable. Theoretically, one could set
        # `pq.Config(dtype=np.float32)`, but this might not always work.
        from jax import config

        config.update("jax_enable_x64", True)

        from piquasso._math.jax.permanent import permanent_with_reduction

        self.np = jnp
        self._scipy = jax.scipy
        self.fallback_np = np
        self.forward_pass_np = jnp
        self.block_diag = jax.scipy.linalg.block_diag
        self.block = jnp.block
        self.logm = partial(jax.scipy.linalg.funm, func=jnp.log)
        self.expm = jax.scipy.linalg.expm
        self.powm = jnp.linalg.matrix_power
        self.sqrtm = jax.scipy.linalg.sqrtm
        self.svd = jnp.linalg.svd
        self.schur = jax.scipy.linalg.schur
        self.permanent = permanent_with_reduction

    def preprocess_input_for_custom_gradient(self, value):
        return value

    def assign(self, array, index, value):
        return array.at[index].set(value)

    def scatter(self, indices, updates, shape):
        embedded_matrix = self.np.zeros(shape, dtype=updates[0].dtype)
        indices_array = self.np.array(indices)
        composite_index = tuple([indices_array[:, i] for i in range(len(shape))])

        return embedded_matrix.at[composite_index].set(self.np.array(updates))

    def polar(self, a, side):
        # NOTE: The default QDWH algorithm does not support left polar decomposition
        # in `jax.scipy.linalg.polar`, so we have to switch to SVD.
        return self._scipy.linalg.polar(a, side, method="svd")
