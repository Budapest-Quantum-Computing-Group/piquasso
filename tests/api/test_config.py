#
# Copyright 2021-2022 Budapest Quantum Computing Group
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
    )

    assert config1 == config1
    assert config3 == config3
    assert config1 == config2
    assert not (config1 == config3)


def test_Config_with_overriding_defaults():
    """
    NOTE: This test basically tests Python itself, but it is left here for us to
    remember that the `Config` class defaults need to be able to overridden for any
    plugin which might want to use e.g. different permanent or hafnian calculation.
    """

    def plugin_permanent_function():
        return 42

    def plugin_loop_hafnian_function():
        return 43

    class PluginConfig(pq.api.config.Config):
        def __init__(
            self,
            permanent_function=plugin_permanent_function,
            loop_hafnian_function=plugin_loop_hafnian_function,
            **kwargs
        ) -> None:
            super().__init__(
                permanent_function=permanent_function,
                loop_hafnian_function=loop_hafnian_function,
                **kwargs,
            )

    plugin_config = PluginConfig()

    assert plugin_config.permanent_function is plugin_permanent_function
    assert plugin_config.loop_hafnian_function is plugin_loop_hafnian_function

    assert plugin_config.permanent_function() == 42
    assert plugin_config.loop_hafnian_function() == 43


def test_Config_subclass_defaults_can_be_overridden_by_user():
    """
    NOTE: This test basically tests Python itself, but it is left here for us to
    remember that the `Config` class defaults need to be able to overridden for any
    plugin which might want to use e.g. different permanent or hafnian calculation.
    """

    def plugin_permanent_function():
        return 42

    def user_defined_permanent_function():
        return 44

    class PluginConfig(pq.api.config.Config):
        def __init__(
            self, permanent_function=plugin_permanent_function, **kwargs
        ) -> None:
            super().__init__(
                permanent_function=permanent_function,
                **kwargs,
            )

    plugin_config = PluginConfig(permanent_function=user_defined_permanent_function)

    assert plugin_config.permanent_function is not plugin_permanent_function

    assert plugin_config.permanent_function is user_defined_permanent_function
