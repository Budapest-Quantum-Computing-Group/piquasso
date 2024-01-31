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

        pq.Q(0) | pq.Fourier()
        pq.Q(1) | pq.Fourier()

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
    pq.Q(0) | pq.Fourier()
    pq.Q(1) | pq.Fourier()

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
            dtype=np.float32,
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
use_torontonian=True, cutoff=6, measurement_cutoff=4, dtype=np.float32)
)

result = simulator.execute(program, shots=100)
"""
    )


def test_Kerr_scalar_parameter_on_multimode_code_generation():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
        pq.Q() | pq.DensityMatrix(ket=(0, 1), bra=(0, 1)) / 4

        pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(0, 2)) / 4
        pq.Q() | pq.DensityMatrix(ket=(2, 0), bra=(2, 0)) / 2

        pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(2, 0)) * np.sqrt(1 / 8)
        pq.Q() | pq.DensityMatrix(ket=(2, 0), bra=(0, 2)) * np.sqrt(1 / 8)

        pq.Q(0) | pq.Kerr(xi=0.2)
        pq.Q(1) | pq.Kerr(xi=0.2)

    simulator = pq.FockSimulator(d=2)

    code = pq.as_code(program, simulator, shots=10)
    code_is_executable(code)

    assert (
        code
        == f"""\
import numpy as np
import piquasso as pq


with pq.Program() as program:
    pq.Q() | pq.Vacuum()
    pq.Q() | pq.DensityMatrix(ket=(0, 1), bra=(0, 1), coefficient=0.25)
    pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(0, 2), coefficient=0.25)
    pq.Q() | pq.DensityMatrix(ket=(2, 0), bra=(2, 0), coefficient=0.5)
    pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(2, 0), coefficient=0.3535533905932738)
    pq.Q() | pq.DensityMatrix(ket=(2, 0), bra=(0, 2), coefficient=0.3535533905932738)
    pq.Q(0) | pq.Kerr(xi=0.2)
    pq.Q(1) | pq.Kerr(xi=0.2)

simulator = pq.{pq.FockSimulator.__name__}(d=2)

result = simulator.execute(program, shots=10)
"""
    )


def test_Kerr_vector_parameter_code_generation():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
        pq.Q() | pq.DensityMatrix(ket=(0, 1), bra=(0, 1)) / 4

        pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(0, 2)) / 4
        pq.Q() | pq.DensityMatrix(ket=(2, 0), bra=(2, 0)) / 2

        pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(2, 0)) * np.sqrt(1 / 8)
        pq.Q() | pq.DensityMatrix(ket=(2, 0), bra=(0, 2)) * np.sqrt(1 / 8)

        pq.Q(0) | pq.Kerr(xi=-0.1)
        pq.Q(1) | pq.Kerr(xi=0.2)

    simulator = pq.FockSimulator(d=2)

    code = pq.as_code(program, simulator, shots=10)
    code_is_executable(code)

    assert (
        code
        == f"""\
import numpy as np
import piquasso as pq


with pq.Program() as program:
    pq.Q() | pq.Vacuum()
    pq.Q() | pq.DensityMatrix(ket=(0, 1), bra=(0, 1), coefficient=0.25)
    pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(0, 2), coefficient=0.25)
    pq.Q() | pq.DensityMatrix(ket=(2, 0), bra=(2, 0), coefficient=0.5)
    pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(2, 0), coefficient=0.3535533905932738)
    pq.Q() | pq.DensityMatrix(ket=(2, 0), bra=(0, 2), coefficient=0.3535533905932738)
    pq.Q(0) | pq.Kerr(xi=-0.1)
    pq.Q(1) | pq.Kerr(xi=0.2)

simulator = pq.{pq.FockSimulator.__name__}(d=2)

result = simulator.execute(program, shots=10)
"""
    )


def test_CubicPhase_vector_parameter_code_generation():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
        pq.Q() | pq.DensityMatrix(ket=(0, 1), bra=(0, 1)) / 4

        pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(0, 2)) / 4
        pq.Q() | pq.DensityMatrix(ket=(2, 0), bra=(2, 0)) / 2

        pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(2, 0)) * np.sqrt(1 / 8)
        pq.Q() | pq.DensityMatrix(ket=(2, 0), bra=(0, 2)) * np.sqrt(1 / 8)

        pq.Q(0) | pq.CubicPhase(gamma=-0.1)
        pq.Q(1) | pq.CubicPhase(gamma=0.2)

    simulator = pq.FockSimulator(d=2)

    code = pq.as_code(program, simulator, shots=10)
    code_is_executable(code)

    assert (
        code
        == f"""\
import numpy as np
import piquasso as pq


with pq.Program() as program:
    pq.Q() | pq.Vacuum()
    pq.Q() | pq.DensityMatrix(ket=(0, 1), bra=(0, 1), coefficient=0.25)
    pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(0, 2), coefficient=0.25)
    pq.Q() | pq.DensityMatrix(ket=(2, 0), bra=(2, 0), coefficient=0.5)
    pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(2, 0), coefficient=0.3535533905932738)
    pq.Q() | pq.DensityMatrix(ket=(2, 0), bra=(0, 2), coefficient=0.3535533905932738)
    pq.Q(0) | pq.CubicPhase(gamma=-0.1)
    pq.Q(1) | pq.CubicPhase(gamma=0.2)

simulator = pq.{pq.FockSimulator.__name__}(d=2)

result = simulator.execute(program, shots=10)
"""
    )


def test_CubicPhase_scalar_parameter_on_multimode_code_generation():
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
        pq.Q() | pq.DensityMatrix(ket=(0, 1), bra=(0, 1)) / 4

        pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(0, 2)) / 4
        pq.Q() | pq.DensityMatrix(ket=(2, 0), bra=(2, 0)) / 2

        pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(2, 0)) * np.sqrt(1 / 8)
        pq.Q() | pq.DensityMatrix(ket=(2, 0), bra=(0, 2)) * np.sqrt(1 / 8)

        pq.Q(0) | pq.CubicPhase(gamma=0.2)
        pq.Q(1) | pq.CubicPhase(gamma=0.2)

    simulator = pq.FockSimulator(d=2)

    code = pq.as_code(program, simulator, shots=10)
    code_is_executable(code)

    assert (
        code
        == f"""\
import numpy as np
import piquasso as pq


with pq.Program() as program:
    pq.Q() | pq.Vacuum()
    pq.Q() | pq.DensityMatrix(ket=(0, 1), bra=(0, 1), coefficient=0.25)
    pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(0, 2), coefficient=0.25)
    pq.Q() | pq.DensityMatrix(ket=(2, 0), bra=(2, 0), coefficient=0.5)
    pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(2, 0), coefficient=0.3535533905932738)
    pq.Q() | pq.DensityMatrix(ket=(2, 0), bra=(0, 2), coefficient=0.3535533905932738)
    pq.Q(0) | pq.CubicPhase(gamma=0.2)
    pq.Q(1) | pq.CubicPhase(gamma=0.2)

simulator = pq.{pq.FockSimulator.__name__}(d=2)

result = simulator.execute(program, shots=10)
"""
    )
