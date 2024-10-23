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
import matplotlib.pyplot as plt

import piquasso as pq

np.set_printoptions(suppress=True, linewidth=200)


def main():
    d = 5
    shots = 20000
    mean_photon_number = 1.2

    adjacency_matrix = np.array(
        [
            [0, 1, 1, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 1, 0, 1, 1],
            [0, 0, 1, 0, 1],
            [1, 1, 1, 1, 0],
        ]
    )

    # NOTE: In SF the measurement cutoff is 5, couldn't be changed, and in PQ it
    # corresponds to `measurement_cutoff=6`.
    simulator = pq.GaussianSimulator(d=d, config=pq.Config(measurement_cutoff=6))

    with pq.Program() as program:
        pq.Q() | pq.Graph(adjacency_matrix, mean_photon_number=mean_photon_number)

        pq.Q(0, 1, 2) | pq.ParticleNumberMeasurement()

    result = simulator.execute(program, shots=shots)

    samples = result.samples
    state = result.state
    reduced_state = state.reduced((0, 1, 2))

    samples_as_list = [tuple(x) for x in samples]

    sample_set = set(samples_as_list)

    f_obs = []
    f_exp = []

    for sample in sample_set:
        count = samples_as_list.count(sample)
        if count < 10:
            continue

        f_obs.append(count / shots)
        f_exp.append(reduced_state.get_particle_detection_probability(sample).real)

    f_obs = np.array(f_obs) / np.sum(f_obs)
    f_exp = np.array(f_exp) / np.sum(f_exp)

    plt.bar(np.arange(len(f_obs)), f_exp, fc=(0.5, 0, 0.5, 0.5), label="Expected")
    plt.bar(np.arange(len(f_obs)), f_obs, fc=(0, 1, 0, 0.5), label="Observed")

    plt.xlim(-1, len(f_obs))

    plt.legend()

    plt.get_current_fig_manager().full_screen_toggle()

    plt.show()


if __name__ == "__main__":
    main()
