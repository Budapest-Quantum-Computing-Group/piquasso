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

"""
CVQNN Module
============

Some tools for simulating CVQNN (Continuous-Variable Quantum Neural Networks) are also
available in Piquasso.
"""

from typing import List, Optional

import numpy as np

from dataclasses import dataclass

from piquasso.api.program import Program
from piquasso.api.instruction import Instruction
from piquasso.api.exceptions import CVQNNException
from piquasso.instructions.preparations import Vacuum
from piquasso.instructions.gates import (
    Phaseshifter,
    Beamsplitter,
    Squeezing,
    Displacement,
    Kerr,
)


@dataclass
class _InterferometerParameters:
    thetas: np.ndarray
    phis: np.ndarray
    final_phis: np.ndarray


@dataclass
class _LayerParameters:
    first_interferometer: _InterferometerParameters
    squeezing_parameters: np.ndarray
    second_interferometer: _InterferometerParameters
    displacement_amplitudes: np.ndarray
    displacement_angles: np.ndarray
    kerr_parameters: np.ndarray


def get_cvqnn_weight_indices(d):
    M = _get_number_of_interferometer_parameters(d)

    K = _get_number_of_single_layer_parameters(d)

    index_range = np.arange(K)

    return [
        index_range[:M],
        index_range[M : M + d],
        index_range[M + d : 2 * M + d],
        index_range[2 * M + d : 2 * M + 2 * d],
        index_range[2 * M + 2 * d : 2 * M + 3 * d],
        index_range[2 * M + 3 * d :],
    ]


def generate_random_cvqnn_weights(
    layer_count: int,
    d: int,
    active_var: float = 0.01,
    passive_var: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generates random CVQNN weights for :func:`create_program`.

    Args:
        layer_count (int): The number of layers.
        d (int): Number of modes.
        active_var (float, optional): Active gate variance. Defaults to 0.01.
        passive_var (float, optional): Passive gate variance. Defaults to 0.1.

    Returns:
        np.ndarray: An array of weights.
    """
    if rng is None:
        rng = np.random.default_rng()

    M = _get_number_of_interferometer_parameters(d)

    first_interferometer = rng.normal(size=[layer_count, M], scale=passive_var)
    squeezing_parameters = rng.normal(size=[layer_count, d], scale=active_var)
    second_interferometer = rng.normal(size=[layer_count, M], scale=passive_var)
    displacement_amplitudes = rng.normal(size=[layer_count, d], scale=active_var)
    displacement_angles = rng.normal(size=[layer_count, d], scale=passive_var)
    kerr_parameters = rng.normal(size=[layer_count, d], scale=active_var)

    return np.concatenate(
        [
            first_interferometer,
            squeezing_parameters,
            second_interferometer,
            displacement_amplitudes,
            displacement_angles,
            kerr_parameters,
        ],
        axis=1,
    )


def create_layers(weights: np.ndarray) -> Program:
    """Creates a subprogram from the specified weights.

    Example usage::

        import piquasso as pq

        weights = pq.cvqnn.generate_random_cvqnn_weights(layer_count=10, d=5)

        cvnn_layers = pq.cvqnn.create_program(weights)

        with pq.Program() as program:
            for i in range(d):
                pq.Q(i) | pq.Displacement(r=0.1)

            pq.Q() | cvnn_layers

    Args:
        weights (np.ndarray): The CVQNN circuit weights.

    Returns:
        Program: The program of the CVQNN circuit, ready to be executed.
    """

    d = get_number_of_modes(weights.shape[1])
    layer_count = weights.shape[0]

    instructions = []

    for k in range(layer_count):
        layer_parameters = _parse_layer(weights[k], d)
        instructions.extend(_instructions_from_layer(layer_parameters, d))

    return Program(instructions)


def create_program(weights: np.ndarray) -> Program:
    """Creates a `Program` from the specified weights, with vacuum initial state.

    Args:
        weights (np.ndarray): The CVQNN circuit weights.

    Returns:
        Program: The program of the CVQNN circuit, ready to be executed.
    """

    cvqnn_layers = create_layers(weights)

    program = Program(instructions=[Vacuum()] + cvqnn_layers.instructions)

    return program


def _get_number_of_single_layer_parameters(d):
    return 2 * _get_number_of_interferometer_parameters(d) + 4 * d


def get_number_of_modes(number_of_parameters: int) -> int:
    """Returns the number of modes given the number of parameters in a CVQNN layer.

    Args:
        number_of_parameters (int): The number of CVQNN parameters per layer.

    Raises:
        CVQNNException:
            If the number of parameters does not correspond to any number of modes

    Returns:
        int: The number of modes.
    """
    if number_of_parameters == 6:
        return 1

    d = 2

    while _get_number_of_single_layer_parameters(d) < number_of_parameters:
        d += 1

    if not _get_number_of_single_layer_parameters(d) == number_of_parameters:
        raise CVQNNException(
            f"Invalid number of parameters specified: '{number_of_parameters}'."
        )

    return d


def _get_number_of_interferometer_parameters(d):
    return int(d * (d - 1)) + max(1, d - 1)


def _parse_interferometer(weights: np.ndarray, d: int) -> _InterferometerParameters:
    return _InterferometerParameters(
        thetas=weights[: d * (d - 1) // 2],
        phis=weights[d * (d - 1) // 2 : d * (d - 1)],
        final_phis=weights[-d + 1 :],
    )


def _parse_layer(weights: np.ndarray, d: int) -> _LayerParameters:
    M = _get_number_of_interferometer_parameters(d)

    return _LayerParameters(
        _parse_interferometer(weights[:M], d),
        weights[M : M + d],
        _parse_interferometer(weights[M + d : 2 * M + d], d),
        weights[2 * M + d : 2 * M + 2 * d],
        weights[2 * M + 2 * d : 2 * M + 3 * d],
        weights[2 * M + 3 * d : 2 * M + 4 * d],
    )


def _instructions_from_interferometer(
    interferometer_parameters: _InterferometerParameters, d: int
) -> List[Instruction]:
    instructions = []

    thetas = interferometer_parameters.thetas
    phis = interferometer_parameters.phis
    final_phis = interferometer_parameters.final_phis

    if d == 1:
        return [Phaseshifter(final_phis[0]).on_modes(0)]

    n = 0
    indices = [list(range(0, d - 1, 2)), list(range(1, d - 1, 2))]

    for j in range(d):
        for k in indices[j % 2]:
            instructions.append(
                Beamsplitter(theta=thetas[n], phi=phis[n]).on_modes(k, k + 1)
            )
            n += 1

    instructions.extend(
        [Phaseshifter(final_phis[i]).on_modes(i) for i in range(max(1, d - 1))]
    )

    return instructions


def _instructions_from_layer(
    layer_parameters: _LayerParameters, d: int
) -> List[Instruction]:
    instructions = []

    instructions.extend(
        _instructions_from_interferometer(layer_parameters.first_interferometer, d)
    )

    squeezing_parameters = layer_parameters.squeezing_parameters
    instructions.extend(
        [Squeezing(squeezing_parameters[i]).on_modes(i) for i in range(d)]
    )

    instructions.extend(
        _instructions_from_interferometer(layer_parameters.second_interferometer, d)
    )

    displacement_amplitudes = layer_parameters.displacement_amplitudes
    displacement_angles = layer_parameters.displacement_angles
    instructions.extend(
        [
            Displacement(displacement_amplitudes[i], displacement_angles[i]).on_modes(i)
            for i in range(d)
        ]
    )

    kerr_parameters = layer_parameters.kerr_parameters
    instructions.extend([Kerr(kerr_parameters[i]).on_modes(i) for i in range(d)])

    return instructions
