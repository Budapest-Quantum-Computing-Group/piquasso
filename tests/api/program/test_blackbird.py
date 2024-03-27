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


def test_loads_blackbird_parses_operations():
    blackbird_code = """name StateTeleportation
        version 1.0

        BSgate(0.7853981633974483, 0) | [1, 2]
        Rgate(0.7853981633974483) | 1
        Vgate(0.5) | 1
        """

    program = pq.Program()

    program.loads_blackbird(blackbird_code)

    assert len(program.instructions) == 3

    assert program.instructions[0] == pq.Beamsplitter(
        theta=np.pi / 4, phi=0.0
    ).on_modes(1, 2)
    assert program.instructions[1] == pq.Phaseshifter(phi=np.pi / 4).on_modes(1)
    assert program.instructions[2] == pq.CubicPhase(gamma=0.5).on_modes(1)


def test_loads_blackbird_parses_operations_with_default_arguments():
    blackbird_code = """name StateTeleportation
        version 1.0

        BSgate() | [1, 2]
        Rgate(0.7853981633974483) | 1
        """

    program = pq.Program()

    program.loads_blackbird(blackbird_code)

    assert len(program.instructions) == 2

    assert program.instructions[0] == pq.Beamsplitter(
        theta=np.pi / 4, phi=0.0
    ).on_modes(1, 2)
    assert program.instructions[1] == pq.Phaseshifter(phi=np.pi / 4).on_modes(1)


def test_loads_blackbird_parses_operations_with_classes_registered_separately():
    blackbird_code = """name StateTeleportation
        version 1.0

        BSgate(0.7853981633974483, 0) | [1, 2]
        Rgate(0.7853981633974483) | 1
        """

    class Beamsplitter(pq.Instruction):
        pass

    program = pq.Program()

    program.loads_blackbird(blackbird_code)

    assert len(program.instructions) == 2

    assert program.instructions[0].__class__ is Beamsplitter

    # Teardown
    pq.Instruction.set_subclass(pq.Beamsplitter)
    assert pq.Instruction.get_subclass("Beamsplitter") is pq.Beamsplitter


def test_loads_blackbird_preserves_exising_operations():
    blackbird_code = """name StateTeleportation
        version 1.0

        BSgate(0.7853981633974483, 0) | [1, 2]
        Rgate(0.7853981633974483) | 1
        """

    program = pq.Program()

    squeezing = pq.Squeezing(r=np.log(2), phi=np.pi / 2)

    with program:
        pq.Q(0) | squeezing

    program.loads_blackbird(blackbird_code)

    assert len(program.instructions) == 3

    assert program.instructions[0] == squeezing
    assert program.instructions[1] == pq.Beamsplitter(
        theta=np.pi / 4, phi=0.0
    ).on_modes(1, 2)
    assert program.instructions[2] == pq.Phaseshifter(phi=np.pi / 4).on_modes(1)


def test_loads_blackbird_with_execution(gaussian_state_assets):
    blackbird_code = """name StateTeleportation
        version 1.0

        BSgate(0.7853981633974483, 3.141592653589793) | [1, 2]
        Rgate(0.7853981633974483) | 1
        """

    program = pq.Program()

    simulator = pq.GaussianSimulator(d=3)

    squeezing = pq.Squeezing(r=np.log(2), phi=np.pi / 2)

    with program:
        pq.Q(1) | squeezing

    program.loads_blackbird(blackbird_code)

    state = simulator.execute(program).state

    expected_state = gaussian_state_assets.load()

    assert state == expected_state


def test_load_blackbird_from_file_with_execution(gaussian_state_assets, tmpdir):
    blackbird_code = """name StateTeleportation
        version 1.0

        BSgate(0.7853981633974483, 3.141592653589793) | [1, 2]
        Rgate(0.7853981633974483) | 1
        """

    blackbird_file = tmpdir.join("example-blackbird-code.xbb")

    blackbird_file.write(blackbird_code)

    program = pq.Program()

    simulator = pq.GaussianSimulator(d=3)

    squeezing = pq.Squeezing(r=np.log(2), phi=np.pi / 2)

    with program:
        pq.Q(1) | squeezing

    program.load_blackbird(blackbird_file.strpath)

    state = simulator.execute(program).state

    expected_state = gaussian_state_assets.load()

    assert state == expected_state
