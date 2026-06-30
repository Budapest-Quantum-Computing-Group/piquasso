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

from fractions import Fraction
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, cast

import numpy as np

from piquasso.api.state import State
from piquasso.api.branch import Branch
from piquasso.api.instruction import Instruction
from piquasso.api.exceptions import InvalidParameter


def create_imperfect_particle_number_measurement(
    particle_number_measurement_simulation_step: Callable,
) -> Callable:
    def imperfect_particle_number_measurement(
        state: State, instruction: Instruction, shots: Optional[int]
    ) -> List[Branch]:
        branches = particle_number_measurement_simulation_step(
            state, instruction, shots
        )

        detector_efficiency_matrix: np.ndarray = np.asarray(
            instruction.params["detector_efficiency_matrix"],
            dtype=state._config.dtype,
        )

        detected_frequencies: Dict[Tuple[int, ...], Fraction] = {}
        imperfect_branches: List[Branch] = []

        for branch in branches:
            branch_frequencies = _get_imperfect_branch_frequencies(
                branch=branch,
                shots=shots,
                detector_efficiency_matrix=detector_efficiency_matrix,
                rng=state._config.rng,
            )

            for detected_outcome, frequency in branch_frequencies.items():
                if frequency == 0:
                    continue

                if branch.state is None:
                    detected_frequencies[detected_outcome] = (
                        detected_frequencies.get(detected_outcome, Fraction(0, 1))
                        + frequency
                    )
                else:
                    imperfect_branches.append(
                        Branch(
                            state=branch.state.copy(),
                            outcome=detected_outcome,
                            frequency=frequency,
                        )
                    )

        imperfect_branches.extend(
            Branch(
                state=None,
                outcome=outcome,
                frequency=frequency,
            )
            for outcome, frequency in detected_frequencies.items()
        )

        return imperfect_branches

    return imperfect_particle_number_measurement


def _get_imperfect_branch_frequencies(
    branch: Branch,
    shots: Optional[int],
    detector_efficiency_matrix: np.ndarray,
    rng: np.random.Generator,
) -> Dict[Tuple[int, ...], Fraction]:
    actual_outcome = cast(Tuple[int, ...], branch.outcome)

    if shots is None:
        return {
            outcome: branch.frequency * probability
            for outcome, probability in _get_detected_outcome_probabilities(
                actual_outcome=actual_outcome,
                detector_efficiency_matrix=detector_efficiency_matrix,
            ).items()
        }

    multiplicity = (branch.frequency * shots).numerator

    return {
        outcome: Fraction(count, shots)
        for outcome, count in _sample_detected_outcomes(
            actual_outcome=actual_outcome,
            multiplicity=multiplicity,
            detector_efficiency_matrix=detector_efficiency_matrix,
            rng=rng,
        ).items()
    }


def _get_detected_outcome_probabilities(
    actual_outcome: Tuple[int, ...],
    detector_efficiency_matrix: np.ndarray,
) -> Dict[Tuple[int, ...], Fraction]:
    probabilities_by_mode = _get_probabilities_by_mode(
        actual_outcome=actual_outcome,
        detector_efficiency_matrix=detector_efficiency_matrix,
    )

    number_of_detectable_counts = detector_efficiency_matrix.shape[0]
    detected_outcome_probabilities: Dict[Tuple[int, ...], Fraction] = {}

    for detected_outcome in product(
        range(number_of_detectable_counts),
        repeat=len(actual_outcome),
    ):
        probability = Fraction(1, 1)

        for mode, detected_count in enumerate(detected_outcome):
            probability *= Fraction(
                float(probabilities_by_mode[mode][detected_count])
            ).limit_denominator()

        if probability != 0:
            detected_outcome_probabilities[detected_outcome] = probability

    return detected_outcome_probabilities


def _get_probabilities_by_mode(
    actual_outcome: Tuple[int, ...],
    detector_efficiency_matrix: np.ndarray,
) -> List[np.ndarray]:
    number_of_actual_counts = detector_efficiency_matrix.shape[1]

    for actual_count in actual_outcome:
        if actual_count >= number_of_actual_counts:
            raise InvalidParameter(
                f"The detector efficiency matrix has no value for photon count "
                f"{actual_count}. Increase the number of columns."
            )

    return [
        detector_efficiency_matrix[:, actual_count] for actual_count in actual_outcome
    ]


def _sample_detected_outcomes(
    actual_outcome: Tuple[int, ...],
    multiplicity: int,
    detector_efficiency_matrix: np.ndarray,
    rng: np.random.Generator,
) -> Dict[Tuple[int, ...], int]:
    probabilities_by_mode = _get_probabilities_by_mode(
        actual_outcome=actual_outcome,
        detector_efficiency_matrix=detector_efficiency_matrix,
    )

    number_of_detectable_counts = detector_efficiency_matrix.shape[0]

    detected_counts_by_mode = [
        rng.choice(
            number_of_detectable_counts,
            size=multiplicity,
            p=probabilities,
        )
        for probabilities in probabilities_by_mode
    ]

    detected_outcomes = np.column_stack(detected_counts_by_mode)

    unique_outcomes, first_indices, counts = np.unique(
        detected_outcomes,
        axis=0,
        return_index=True,
        return_counts=True,
    )

    order = np.argsort(first_indices)

    return {
        tuple(int(value) for value in outcome): int(count)
        for outcome, count in zip(unique_outcomes[order], counts[order])
    }
