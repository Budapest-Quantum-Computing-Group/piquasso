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

import pytest

import numpy as np

import piquasso as pq

from scipy.stats import chisquare, unitary_group


@pytest.mark.monkey
def test_gaussian_boson_sampling_chi_square_hypothesis_test():
    d = 3
    shots = 10000

    with pq.Program() as gaussian_boson_sampling:
        for i in range(d):
            pq.Q(i) | pq.Squeezing(np.random.rand() * 0.1) | pq.Displacement(
                np.random.normal(loc=0.0, scale=0.5)
            )

        pq.Q(all) | pq.Interferometer(unitary_group.rvs(d))

        pq.Q(all) | pq.ParticleNumberMeasurement()

    simulator = pq.GaussianSimulator(
        d=d,
        connector=pq.NumpyConnector(),
        config=pq.Config(
            measurement_cutoff=10,
        ),
    )
    result = simulator.execute(gaussian_boson_sampling, shots=shots)
    samples = result.samples
    state = result.state

    samples_as_list = [tuple(x) for x in samples]

    sample_set = set(samples_as_list)

    f_obs = []
    f_exp = []

    for sample in sample_set:
        f_obs.append(samples_as_list.count(sample) / shots)
        f_exp.append(state.get_particle_detection_probability(sample).real)

    f_obs = np.array(f_obs) / np.sum(f_obs)
    f_exp = np.array(f_exp) / np.sum(f_exp)

    test = chisquare(f_obs, f_exp)

    assert test.pvalue >= 0.05, (
        "Chi-square hypothesis test failed: the GBS samples differ from the expected "
        "distribution."
    )


@pytest.mark.monkey
def test_threshold_gaussian_boson_sampling_chi_square_hypothesis_test():
    d = 3
    shots = 10000

    with pq.Program() as gaussian_boson_sampling:
        for i in range(d):
            pq.Q(i) | pq.Squeezing(np.random.rand() * 0.1) | pq.Displacement(
                np.random.normal(loc=0.0, scale=0.5)
            )

        pq.Q(all) | pq.Interferometer(unitary_group.rvs(d))

        pq.Q(all) | pq.ThresholdMeasurement()

    simulator = pq.GaussianSimulator(
        d=d,
        connector=pq.NumpyConnector(),
        config=pq.Config(
            measurement_cutoff=10,
        ),
    )
    result = simulator.execute(gaussian_boson_sampling, shots=shots)
    samples = result.samples
    state = result.state

    samples_as_list = [tuple(x) for x in samples]

    sample_set = set(samples_as_list)

    f_obs = []
    f_exp = []

    for sample in sample_set:
        f_obs.append(samples_as_list.count(sample) / shots)
        f_exp.append(state.get_particle_detection_probability(sample).real)

    f_obs = np.array(f_obs) / np.sum(f_obs)
    f_exp = np.array(f_exp) / np.sum(f_exp)

    test = chisquare(f_obs, f_exp)

    assert test.pvalue >= 0.05, (
        "Chi-square hypothesis test failed: the GBS samples differ from the expected "
        "distribution."
    )


@pytest.mark.monkey
def test_threshold_gaussian_boson_sampling_torontonian_chi_square_hypothesis_test():
    d = 3
    shots = 10000

    with pq.Program() as gaussian_boson_sampling:
        for i in range(d):
            pq.Q(i) | pq.Squeezing(np.random.rand() * 0.1) | pq.Displacement(
                np.random.normal(loc=0.0, scale=0.5)
            )

        pq.Q(all) | pq.Interferometer(unitary_group.rvs(d))

        pq.Q(all) | pq.ThresholdMeasurement()

    simulator = pq.GaussianSimulator(
        d=d,
        connector=pq.NumpyConnector(),
        config=pq.Config(
            measurement_cutoff=10,
            use_torontonian=True,
        ),
    )
    result = simulator.execute(gaussian_boson_sampling, shots=shots)
    samples = result.samples
    state = result.state

    samples_as_list = [tuple(x) for x in samples]

    sample_set = set(samples_as_list)

    f_obs = []
    f_exp = []

    for sample in sample_set:
        f_obs.append(samples_as_list.count(sample) / shots)
        f_exp.append(state.get_particle_detection_probability(sample).real)

    f_obs = np.array(f_obs) / np.sum(f_obs)
    f_exp = np.array(f_exp) / np.sum(f_exp)

    test = chisquare(f_obs, f_exp)

    assert test.pvalue >= 0.05, (
        "Chi-square hypothesis test failed: the GBS samples differ from the expected "
        "distribution."
    )
