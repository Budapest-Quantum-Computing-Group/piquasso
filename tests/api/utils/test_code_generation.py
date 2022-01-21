#
# Copyright 2021 Budapest Quantum Computing Group
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


def code_is_executable(code):
    exec(code)


def test_empty_config_code_generation():
    # This is tested separately as we can not test it using the public interface of
    # piquasso
    config = pq.Config()
    code = config._as_code()
    code_is_executable(code)

    assert code == "pq.Config()"


def test_empty_code_generation():
    with pq.Program() as program:
        pass

    simulator = pq.GaussianSimulator(d=2)

    code = pq.as_code(program, simulator)
    code_is_executable(code)

    assert (
        code
        == f"""\
import numpy as np
import piquasso as pq


with pq.Program() as program:
    pass

simulator = pq.{pq.GaussianSimulator.__name__}(d=2)

result = simulator.execute(program, shots=1)
"""
    )


def test_complicated_code_generation():
    with pq.Program() as program:
        pq.Q(0) | pq.Squeezing(r=0.5, phi=0)
        pq.Q(1) | pq.Squeezing(r=0.5, phi=0)
        pq.Q(0, 1) | pq.Beamsplitter(theta=1, phi=0.7853981633974483)

        pq.Q(all) | pq.Fourier()

    simulator = pq.GaussianSimulator(d=2, config=pq.Config(seed_sequence=1, cutoff=7))

    code = pq.as_code(program, simulator, shots=10)
    code_is_executable(code)

    assert (
        code
        == f"""\
import numpy as np
import piquasso as pq


with pq.Program() as program:
    pq.Q(0) | pq.Squeezing(r=0.5, phi=0)
    pq.Q(1) | pq.Squeezing(r=0.5, phi=0)
    pq.Q(0, 1) | pq.Beamsplitter(theta=1, phi=0.7853981633974483)
    pq.Q() | pq.Fourier()

simulator = pq.{pq.GaussianSimulator.__name__}(
    d=2, config=pq.Config(seed_sequence=1, cutoff=7)
)

result = simulator.execute(program, shots=10)
"""
    )


def test_numpy_ndarray_parameter_code_generation():
    with pq.Program() as program:
        pq.Q(0, 1) | pq.GeneraldyneMeasurement(
            detection_covariance=np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            ),
        )

    simulator = pq.GaussianSimulator(d=2)

    code = pq.as_code(program, simulator, shots=420)
    code_is_executable(code)

    assert (
        code
        == f"""\
import numpy as np
import piquasso as pq


with pq.Program() as program:
    pq.Q(0, 1) | pq.GeneraldyneMeasurement(detection_covariance=np.array([[1., 0.],
       [0., 1.]]))

simulator = pq.{pq.GaussianSimulator.__name__}(d=2)

result = simulator.execute(program, shots=420)
"""
    )


def test_full_config_code_generation():
    with pq.Program() as program:
        pq.Q(0) | pq.Fourier()
        pq.Q(0, 1) | pq.Beamsplitter(theta=1, phi=0)
        pq.Q(0, 2) | pq.Squeezing2(r=0.5, phi=0)
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.GaussianSimulator(
        d=3,
        config=pq.Config(
            seed_sequence=0,
            cache_size=64,
            hbar=2.5,
            use_torontonian=True,
            cutoff=6,
            measurement_cutoff=4,
        ),
    )

    code = pq.as_code(program, simulator, shots=100)
    code_is_executable(code)

    assert (
        code
        == f"""\
import numpy as np
import piquasso as pq


with pq.Program() as program:
    pq.Q(0) | pq.Fourier()
    pq.Q(0, 1) | pq.Beamsplitter(theta=1, phi=0)
    pq.Q(0, 2) | pq.Squeezing2(r=0.5, phi=0)
    pq.Q() | pq.ParticleNumberMeasurement()

simulator = pq.{pq.GaussianSimulator.__name__}(
    d=3, config=pq.Config(seed_sequence=0, cache_size=64, hbar=2.5, \
use_torontonian=True, cutoff=6, measurement_cutoff=4)
)

result = simulator.execute(program, shots=100)
"""
    )
