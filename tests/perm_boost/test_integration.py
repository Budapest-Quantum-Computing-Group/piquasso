#
# Copyright 2021-2026 Budapest Quantum Computing Group
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
#
# Portions of this file are based on work by Bence Soóki-Tóth, used with
# permission and originally made available under the MIT License.
#
# Bence Soóki-Tóth. "Efficient calculation of permanent function gradients
# in photonic quantum computing simulations", Eötvös Loránd University, 2025.

from unittest.mock import patch

import numpy as np
import pytest

from jax import numpy as jnp

perm_boost = pytest.importorskip(
    "piquasso.jax_extensions",
    reason="perm_boost C++ module is not compiled",
)

import piquasso as pq  # noqa: E402


def test_permanent_routes_to_perm_boost_when_flag_is_set():
    """When use_perm_boost=True, permanent must call perm_boost.perm."""
    connector = pq.JaxConnector()

    matrix = jnp.array(
        [
            [1.0 + 0j, 0.0 + 0j, 0.0 + 0j],
            [0.0 + 0j, 1.0 + 0j, 0.0 + 0j],
            [0.0 + 0j, 0.0 + 0j, 1.0 + 0j],
        ],
        dtype=jnp.complex128,
    )
    rows = jnp.array([1, 1, 1], dtype=jnp.uint64)
    cols = jnp.array([1, 1, 1], dtype=jnp.uint64)

    with patch(
        "piquasso.jax_extensions.permanent.perm", wraps=perm_boost.perm
    ) as mock_perm:
        connector.permanent(matrix, rows, cols, use_perm_boost=True)

    mock_perm.assert_called_once()


def test_permanent_does_not_route_to_perm_boost_by_default():
    """When use_perm_boost is not passed, perm_boost.perm must NOT be called."""
    connector = pq.JaxConnector()

    matrix = jnp.array(
        [
            [1.0 + 0j, 0.0 + 0j, 0.0 + 0j],
            [0.0 + 0j, 1.0 + 0j, 0.0 + 0j],
            [0.0 + 0j, 0.0 + 0j, 1.0 + 0j],
        ],
        dtype=jnp.complex128,
    )
    rows = jnp.array([1, 1, 1], dtype=jnp.uint64)
    cols = jnp.array([1, 1, 1], dtype=jnp.uint64)

    with patch(
        "piquasso.jax_extensions.permanent.perm", wraps=perm_boost.perm
    ) as mock_perm:
        connector.permanent(matrix, rows, cols)

    mock_perm.assert_not_called()


def test_perm_boost_permanent_matches_jax_permanent():
    """permanent(use_perm_boost=True) and permanent() agree on permanent values."""
    matrix = jnp.array(
        [
            [
                0.62270314 + 0.55117657j,
                -0.0258677 - 0.07171713j,
                0.09597446 - 0.54168404j,
            ],
            [
                0.34756795 - 0.29444499j,
                -0.43514701 + 0.18975153j,
                0.71929752 + 0.22304973j,
            ],
            [
                -0.24500645 - 0.20227626j,
                -0.45222962 - 0.75121057j,
                0.06995606 - 0.3540245j,
            ],
        ],
        dtype=jnp.complex128,
    )
    rows = jnp.array([1, 1, 1], dtype=jnp.uint64)
    cols = jnp.array([1, 1, 1], dtype=jnp.uint64)

    connector = pq.JaxConnector()

    result_default = connector.permanent(matrix, rows, cols)
    result_perm_boost = connector.permanent(matrix, rows, cols, use_perm_boost=True)

    assert np.isclose(
        complex(result_default),
        complex(result_perm_boost),
        rtol=1e-10,
        atol=1e-12,
    ), (
        f"perm_boost result {result_perm_boost} differs from JAX result "
        f"{result_default}"
    )


def test_jax_connector_simulator_pipeline_runs_without_errors():
    """End-to-end smoke test: PureFockSimulator with JaxConnector executes
    a beamsplitter program. Does NOT exercise the perm_boost FFI path --
    that's covered by the unit tests above (notably
    `test_permanent_routes_to_perm_boost_when_flag_is_set` and
    `test_perm_boost_permanent_matches_jax_permanent`). This test ensures the
    JaxConnector integration with the simulator stack does not regress."""
    connector = pq.JaxConnector()

    simulator = pq.PureFockSimulator(
        d=2,
        config=pq.Config(cutoff=4),
        connector=connector,
    )

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 4)

    result = simulator.execute(program)

    probabilities = np.array(result.state.fock_probabilities)

    assert probabilities is not None
    assert np.all(probabilities >= 0.0)
    assert np.isclose(np.sum(probabilities), 1.0, atol=1e-6)
