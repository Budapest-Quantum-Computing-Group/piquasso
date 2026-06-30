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

from typing import Dict, Tuple, List

import numpy as np

from fractions import Fraction

from piquasso._simulators.sampling.state import SamplingState

from piquasso.instructions import gates

from piquasso.api.exceptions import InvalidState, NotImplementedCalculation
from piquasso.api.branch import Branch
from piquasso.api.instruction import Instruction

from piquasso._math.validations import validate_occupation_numbers
from piquasso._simulators.fock.pure.simulation_steps import (
    imperfect_post_select_photons as pure_fock_imperfect_post_select_photons,
)

from .sampling import (
    generate_samples,
    generate_lossy_samples,
    generate_marginal_samples,
    generate_lossy_and_partially_distinguishable_samples,
    is_direct_marginal_sampling_cheaper,
    map_to_original_modes,
)
from piquasso._utils import get_counts


def vacuum(state: SamplingState, instruction: Instruction, shots: int) -> List[Branch]:
    state._occupation_numbers.append(np.zeros(state.d, dtype=int))
    state._coefficients.append(1.0)

    return [Branch(state=state)]


def create(state: SamplingState, instruction: Instruction, shots: int) -> List[Branch]:
    modes = instruction.modes

    for i in range(len(state._occupation_numbers)):
        occupation_numbers = state._occupation_numbers[i].copy()
        occupation_numbers[modes,] += 1
        _validate_or_infer_cutoff(state, occupation_numbers, instruction)
        state._occupation_numbers[i] = occupation_numbers

    return [Branch(state=state)]


def _validate_or_infer_cutoff(
    state: SamplingState,
    occupation_numbers: np.ndarray,
    instruction: Instruction,
) -> None:
    occupation_numbers_tuple = tuple(int(number) for number in occupation_numbers)

    if state._config.validate and len(occupation_numbers) != state.d:
        raise InvalidState(
            f"The occupation numbers '{occupation_numbers_tuple}' are not "
            f"well-defined on '{state.d}' modes: instruction={instruction}"
        )

    if state._config._cutoff_was_explicit:
        if state._config.validate:
            validate_occupation_numbers(
                occupation_numbers_tuple,
                state.d,
                state._config.cutoff,
                context=f" Instruction: {instruction}.",
            )
        return

    required_cutoff = int(np.sum(occupation_numbers)) + 1
    state._config.cutoff = max(state._config.cutoff, required_cutoff)


def state_vector(
    state: SamplingState, instruction: Instruction, shots: int
) -> List[Branch]:
    coefficient = instruction._get_all_params(state._connector)["coefficient"]

    if not state.is_indistinguishable:
        raise NotImplementedCalculation(
            f"The instruction {instruction} is not supported for partially "
            "distinguishable states."
        )

    if "occupation_numbers" in instruction._get_all_params(state._connector):
        occupation_numbers = instruction._get_all_params(state._connector)[
            "occupation_numbers"
        ]
        occupation_numbers = np.rint(occupation_numbers).astype(int)

        _validate_or_infer_cutoff(state, occupation_numbers, instruction)

        state._occupation_numbers.append(occupation_numbers)
        state._coefficients.append(coefficient)

    elif "fock_amplitude_map" in instruction._get_all_params(state._connector):
        for occupation_numbers, amplitude in instruction._get_all_params(
            state._connector
        )["fock_amplitude_map"].items():

            occupation_numbers = np.rint(occupation_numbers).astype(int)

            _validate_or_infer_cutoff(state, occupation_numbers, instruction)

            state._occupation_numbers.append(occupation_numbers)
            state._coefficients.append(coefficient * amplitude)

    return [Branch(state=state)]


def distinguishable_number_state(
    state: SamplingState, instruction: Instruction, shots: int
) -> List[Branch]:
    if state._occupation_numbers:
        raise NotImplementedCalculation(
            f"The instruction {instruction} is not supported for states defined using "
            "multiple 'NumberState' instructions."
        )

    state._occupation_numbers.append(
        np.rint(
            instruction._get_all_params(state._connector)["occupation_numbers"]
        ).astype(int)
    )
    state._coefficients.append(1.0)

    particle_overlap = instruction._get_all_params(state._connector)["particle_overlap"]
    if np.isscalar(particle_overlap) and np.isclose(particle_overlap, 1.0):
        particle_overlap = None
    state._particle_overlap = particle_overlap

    return [Branch(state=state)]


def passive_linear(
    state: SamplingState, instruction: gates._PassiveLinearGate, shots: int
) -> List[Branch]:
    r"""Applies an interferometer to the circuit.

    This can be interpreted as placing another interferometer in the network, just
    before performing the sampling. This instruction is realized by multiplying
    current effective interferometer matrix with new interferometer matrix.

    Do note, that new interferometer matrix works as interferometer matrix on
    qumodes (provided as the arguments) and as an identity on every other mode.
    """
    _apply_matrix_on_modes(
        state=state,
        matrix=instruction._get_passive_block(state._connector, state._config),
        modes=instruction.modes,
    )

    return [Branch(state=state)]


def _apply_matrix_on_modes(
    state: SamplingState, matrix: np.ndarray, modes: Tuple[int, ...]
) -> None:
    connector = state._connector
    np = connector.np
    fallback_np = connector.fallback_np

    embedded = np.identity(len(state.interferometer), dtype=state._config.complex_dtype)

    actual_modes = fallback_np.array(state._get_active_modes())[modes,]

    embedded = connector.assign(
        embedded, fallback_np.ix_(actual_modes, actual_modes), matrix
    )

    state.interferometer = embedded @ state.interferometer


def loss(state: SamplingState, instruction: Instruction, shots: int) -> List[Branch]:
    state.is_lossy = True

    _apply_matrix_on_modes(
        state=state,
        matrix=np.array(
            [[instruction._get_all_params(state._connector)["transmissivity"]]]
        ),
        modes=instruction.modes,
    )

    return [Branch(state=state)]


def uniform_loss(
    state: SamplingState, instruction: Instruction, shots: int
) -> List[Branch]:
    state.is_lossy = True

    connector = state._connector
    transmissivity = instruction._get_all_params(connector)["transmissivity"]
    modes = instruction.modes

    _apply_matrix_on_modes(
        state=state,
        matrix=connector.np.identity(len(modes), dtype=state._config.complex_dtype)
        * transmissivity,
        modes=modes,
    )

    return [Branch(state=state)]


def lossy_interferometer(
    state: SamplingState, instruction: Instruction, shots: int
) -> List[Branch]:
    state.is_lossy = True

    _apply_matrix_on_modes(
        state=state,
        matrix=instruction._get_all_params(state._connector)["matrix"],
        modes=instruction.modes,
    )

    return [Branch(state=state)]


def particle_number_measurement(
    state: SamplingState, instruction: Instruction, shots: int
) -> List[Branch]:
    """
    Simulates a boson sampling using generalized Clifford & Clifford algorithm
    from [Brod, Oszmaniec 2020] see
    `this article <https://arxiv.org/abs/1906.06696>`_ for more details.

    This is a contribution implementation from `theboss`, see
    https://github.com/Tomev-CTP/theboss.

    This method assumes that initial_state is given in the second quantization
    description (mode occupation).

    Generalized Cliffords simulation strategy form [Brod, Oszmaniec 2020] was used
    as it allows effective simulation of broader range of input states than original
    algorithm.
    """

    if (
        state._can_validate_variable(state._coefficients[0])
        and len(state._occupation_numbers) != 1
        and not np.isclose(state._coefficients[0], 1.0)
    ):
        raise NotImplementedCalculation(
            f"The instruction {instruction} is not supported for states defined using "
            "multiple 'NumberState' instructions.\n"
            "If you need this feature to be implemented, please create an issue at "
            "https://github.com/Budapest-Quantum-Computing-Group/piquasso/issues"
        )

    initial_state = state._occupation_numbers[0]

    rng = state._config.rng

    postselected_modes = state._get_postselected_modes()

    postselect_data = (
        postselected_modes,
        state._get_postselected_photons(),
        state._config.max_sample_generation_trials,
    )

    modes = instruction.modes

    marginal_sampling = set(modes) != set(range(state.d))

    singular_values = np.linalg.svd(state.interferometer, compute_uv=False)

    is_ideal_or_uniform_lossy = np.all(np.isclose(singular_values, singular_values[0]))

    if (
        marginal_sampling
        and is_ideal_or_uniform_lossy
        and is_direct_marginal_sampling_cheaper(
            k=len(modes) + len(postselected_modes),
            d=state.total_number_of_modes,
            n=int(np.sum(initial_state)),
            shots=shots,
        )
    ):
        original_modes = map_to_original_modes(modes, postselected_modes)

        samples = generate_marginal_samples(
            initial_state=initial_state,
            interferometer=state.interferometer,
            modes=original_modes,
            shots=shots,
            rng=rng,
            postselect_data=postselect_data,
        )

        binned_samples = get_counts(samples)

        return _create_branches_after_marginal_particle_number_measurement(
            state=state,
            instruction=instruction,
            shots=shots,
            binned_samples=binned_samples,
        )

    common_kwargs = dict(
        input=initial_state,
        interferometer=state.interferometer,
        shots=shots,
        rng=rng,
        postselect_data=postselect_data,
    )

    if not state.is_lossy and not state.is_nonuniformly_partially_distinguishable:
        samples = generate_samples(
            **common_kwargs,
            calculate_permanent_laplace=state._connector.permanent_laplace,
            reject_condition=lambda: False,
            uniform_particle_overlap=state._particle_overlap,
        )

    elif (
        state.is_uniformly_lossy and not state.is_nonuniformly_partially_distinguishable
    ):
        uniform_transmission_probability = singular_values[0] ** 2

        samples = generate_samples(
            **common_kwargs,
            calculate_permanent_laplace=state._connector.permanent_laplace,
            reject_condition=(lambda: rng.uniform() > uniform_transmission_probability),
            uniform_particle_overlap=state._particle_overlap,
        )

    elif not state.is_partially_distinguishable:
        samples = generate_lossy_samples(
            **common_kwargs,
            calculate_permanent_laplace=state._connector.permanent_laplace,
        )
    else:
        samples = generate_lossy_and_partially_distinguishable_samples(
            **common_kwargs,
            particle_overlap_matrix=state._particle_overlap,
            connector=state._connector,
        )

    binned_samples = get_counts(samples)

    if marginal_sampling:
        projected_binned_samples: Dict[Tuple[int, ...], int] = {}

        for outcome, multiplicity in binned_samples.items():
            projected_outcome = tuple(outcome[mode] for mode in modes)
            projected_binned_samples[projected_outcome] = (
                projected_binned_samples.get(projected_outcome, 0) + multiplicity
            )

        binned_samples = projected_binned_samples

        return _create_branches_after_marginal_particle_number_measurement(
            state=state,
            instruction=instruction,
            shots=shots,
            binned_samples=binned_samples,
        )

    branches = [
        Branch(state=None, outcome=outcome, frequency=Fraction(multiplicity, shots))
        for outcome, multiplicity in binned_samples.items()
    ]

    return branches


def _create_branches_after_marginal_particle_number_measurement(
    state: SamplingState, instruction: Instruction, shots: int, binned_samples: dict
) -> List[Branch]:
    modes = instruction.modes

    branches = []

    for outcome, multiplicity in binned_samples.items():
        new_state = state.copy()
        new_state._postselections = {
            **new_state._postselections,
            **{mode: x for mode, x in zip(modes, outcome)},
        }

        branches.append(
            Branch(
                state=new_state,
                outcome=outcome,
                frequency=Fraction(multiplicity, shots),
            )
        )

    return branches


def post_select_photons(
    state: SamplingState, instruction: Instruction, shots: int
) -> List[Branch]:
    modes = instruction.modes

    photon_counts = instruction.params["photon_counts"]

    state._set_postselection(modes, photon_counts)

    return [Branch(state=state)]


imperfect_post_select_photons = pure_fock_imperfect_post_select_photons


def kerr(state: SamplingState, instruction: gates.Kerr, shots: int) -> List[Branch]:
    state._materialize_state_vector()

    np = state._connector.np

    xi = instruction.params["xi"]

    mode = instruction.modes[0]

    for i in range(len(state._occupation_numbers)):
        state._coefficients[i] *= np.exp(
            1j * xi * state._occupation_numbers[i][mode] ** 2
        )

    return [Branch(state=state)]


def cross_kerr(
    state: SamplingState, instruction: gates.CrossKerr, shots: int
) -> List[Branch]:
    state._materialize_state_vector()

    np = state._connector.np

    xi = instruction.params["xi"]

    mode1, mode2 = instruction.modes

    for i in range(len(state._occupation_numbers)):
        state._coefficients[i] *= np.exp(
            1j
            * xi
            * state._occupation_numbers[i][mode1]
            * state._occupation_numbers[i][mode2]
        )

    return [Branch(state=state)]
