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

"""
Script to simulate boson sampling with postselection and display histogram of results.
"""

import piquasso as pq

import numpy as np

from scipy.stats import unitary_group


if __name__ == "__main__":
    unitary = unitary_group.rvs(3, random_state=42)

    input_state = np.array([1, 1, 1], dtype=int)

    n = sum(input_state)
    d = len(input_state)

    N = 10000

    postselect_modes = [1]
    photon_counts = [1]

    instructions = [
        pq.StateVector(input_state),
        pq.Interferometer(unitary),
        pq.PostSelectPhotons(photon_counts=photon_counts).on_modes(*postselect_modes),
        pq.ParticleNumberMeasurement(),
    ]

    simulator = pq.SamplingSimulator(
        d=len(input_state), config=pq.Config(seed_sequence=42, cutoff=9)
    )

    result = simulator.execute_instructions(instructions, shots=N)

    counts = result.get_counts()

    result_without_measurement = simulator.execute_instructions(instructions[:3])
    state = result_without_measurement.state
    norm = state.norm

    for occupation_numbers, count in counts.items():
        probability = (
            state.get_particle_detection_probability(occupation_numbers) / norm
        )
        expected_count = probability * N

        print(
            f"Occupation numbers: {occupation_numbers}, "
            f"Measured count: {count}, "
            f"Expected count: {expected_count:.2f}"
        )
