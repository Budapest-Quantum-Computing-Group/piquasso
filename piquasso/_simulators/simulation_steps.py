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

from typing import Dict, Tuple, List, Callable

import numpy as np

from fractions import Fraction

from piquasso.api.state import State
from piquasso.api.branch import Branch
from piquasso.api.instruction import Instruction
from piquasso.api.exceptions import InvalidParameter


def create_imperfect_particle_number_measurement(
    particle_number_measurement_simulation_step: Callable,
) -> Callable:
    def imperfect_particle_number_measurement(
        state: State, instruction: Instruction, shots: int
    ) -> List[Branch]:
        branches = particle_number_measurement_simulation_step(
            state, instruction, shots
        )

        detector_efficiency_matrix: np.ndarray = np.asarray(
            instruction.params["detector_efficiency_matrix"],
            dtype=state._config.dtype,
        )

        rng = state._config.rng

        detected_counts: Dict[Tuple[int, ...], int] = {}
        imperfect_branches: List[Branch] = []

        for branch in branches:
            multiplicity = (branch.frequency * shots).numerator

            detected_outcomes = _sample_detected_outcomes_for_branch(
                actual_outcome=branch.outcome,
                multiplicity=multiplicity,
                detector_efficiency_matrix=detector_efficiency_matrix,
                rng=rng,
            )

            for detected_outcome, detected_multiplicity in detected_outcomes.items():
                if branch.state is None:
                    detected_counts[detected_outcome] = (
                        detected_counts.get(detected_outcome, 0) + detected_multiplicity
                    )
                else:
                    imperfect_branches.append(
                        Branch(
                            state=branch.state.copy(),
                            outcome=detected_outcome,
                            frequency=Fraction(detected_multiplicity, shots),
                        )
                    )

        imperfect_branches.extend(
            Branch(
                state=None,
                outcome=outcome,
                frequency=Fraction(multiplicity, shots),
            )
            for outcome, multiplicity in detected_counts.items()
        )

        return imperfect_branches

    return imperfect_particle_number_measurement


def _sample_detected_outcomes_for_branch(
    actual_outcome: Tuple[int, ...],
    multiplicity: int,
    detector_efficiency_matrix: np.ndarray,
    rng: np.random.Generator,
) -> Dict[Tuple[int, ...], int]:
    number_of_detectable_counts, number_of_actual_counts = (
        detector_efficiency_matrix.shape
    )

    detected_counts = np.arange(number_of_detectable_counts)

    detected_counts_by_mode = []

    for actual_count in actual_outcome:
        if actual_count >= number_of_actual_counts:
            raise InvalidParameter(
                f"The detector efficiency matrix has no value for photon count "
                f"{actual_count}. Increase the number of columns."
            )

        probabilities = detector_efficiency_matrix[:, actual_count]
        probabilities = probabilities / np.sum(probabilities)

        detected_counts_by_mode.append(
            rng.choice(
                detected_counts,
                size=multiplicity,
                p=probabilities,
            )
        )

    detected_outcomes = np.stack(detected_counts_by_mode, axis=1)
    unique_outcomes, first_indices, counts = np.unique(
        detected_outcomes, axis=0, return_index=True, return_counts=True
    )

    order = np.argsort(first_indices)

    return {
        tuple(int(value) for value in outcome): int(count)
        for outcome, count in zip(unique_outcomes[order], counts[order])
    }
