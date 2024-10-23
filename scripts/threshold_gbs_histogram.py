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


def main():
    d = 5
    shots = 10000
    measurement_cutoff = 6

    # NOTE: In SF the measurement cutoff is 5, couldn't be changed, and in PQ it
    # corresponds to `measurement_cutoff=6`.
    simulator = pq.GaussianSimulator(
        d=d, config=pq.Config(measurement_cutoff=measurement_cutoff)
    )

    with pq.Program() as program:
        for i in range(d):
            pq.Q(i) | pq.Squeezing(r=0.1) | pq.Displacement(r=1)

        pq.Q(0, 1) | pq.Beamsplitter(0.0959408065906761, 0.06786053071484363)
        pq.Q(2, 3) | pq.Beamsplitter(0.7730047654405018, 1.453770233324797)
        pq.Q(1, 2) | pq.Beamsplitter(1.0152680371119776, 1.2863559998816205)
        pq.Q(3, 4) | pq.Beamsplitter(1.3205517879465705, 0.5236836466492961)
        pq.Q(0, 1) | pq.Beamsplitter(4.394480318177715, 4.481575657714487)
        pq.Q(2, 3) | pq.Beamsplitter(2.2300919706807534, 1.5073556513699888)
        pq.Q(1, 2) | pq.Beamsplitter(2.2679037068773673, 1.9550229282085838)
        pq.Q(3, 4) | pq.Beamsplitter(3.340269832485504, 3.289367083610399)

        pq.Q(all) | pq.ThresholdMeasurement()

    result = simulator.execute(program, shots=shots)

    samples = result.samples
    state = result.state

    samples_as_list = [tuple(x) for x in samples]

    sample_set = set(samples_as_list)

    f_obs = []
    f_exp = []

    for sample in sample_set:
        count = samples_as_list.count(sample)
        if count < 10:
            continue

        f_obs.append(count / shots)
        f_exp.append(state.get_threshold_detection_probability(sample).real)

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
