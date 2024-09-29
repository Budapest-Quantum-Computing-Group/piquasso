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

import piquasso as pq


def test_Config_seed_generates_same_output():
    seed_sequence = 42

    mean = np.array([1, 2])
    covariance = np.array(
        [
            [2, -1],
            [-1, 2],
        ]
    )

    config1 = pq.Config(seed_sequence=seed_sequence)
    config2 = pq.Config(seed_sequence=seed_sequence)

    sample = config1.rng.multivariate_normal(mean=mean, cov=covariance)
    reproduced_sample = config2.rng.multivariate_normal(mean=mean, cov=covariance)

    assert np.allclose(sample, reproduced_sample)


def test_eq():
    config1 = pq.Config(seed_sequence=1, cutoff=7)
    config2 = pq.Config(seed_sequence=1, cutoff=7)
    config3 = pq.Config(
        seed_sequence=0,
        cache_size=64,
        hbar=1,
        use_torontonian=True,
        cutoff=6,
        measurement_cutoff=4,
        validate=False,
    )

    assert config1 == config1
    assert config3 == config3
    assert config1 == config2
    assert not (config1 == config3)


def test_as_code_dtype():
    conf_f32 = pq.Config(dtype=np.float32)
    conf_f64 = pq.Config(dtype=np.float64)
    conf_f = pq.Config(dtype=float)

    as_code_f32 = conf_f32._as_code()
    as_code_f64 = conf_f64._as_code()
    as_code_f = conf_f._as_code()

    assert as_code_f32 == "pq.Config(dtype=np.float32)"
    assert as_code_f64 == "pq.Config()"
    assert as_code_f == "pq.Config()"


def test_as_code_validate():
    default = pq.Config()
    default_2 = pq.Config(validate=True)
    no_validate_config = pq.Config(validate=False)

    assert default._as_code() == default_2._as_code()
    assert default._as_code() == "pq.Config()"
    assert no_validate_config._as_code() == "pq.Config(validate=False)"


def test_complex_dtype():
    conf_f32 = pq.Config(dtype=np.float32)
    conf_f64 = pq.Config(dtype=np.float64)
    conf_f = pq.Config(dtype=float)

    assert conf_f32.complex_dtype == np.complex64
    assert conf_f64.complex_dtype == np.complex128
    assert conf_f.complex_dtype == np.complex128


def test_Config_rng_is_shallow_copied():
    config = pq.Config(seed_sequence=123)

    config_copy = config.copy()

    assert config.rng is config_copy.rng
