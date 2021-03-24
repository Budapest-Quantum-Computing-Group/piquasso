#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np
import piquasso as pq


def test_loads_blackbird_parses_operations():
    blackbird_code = \
        """name StateTeleportation
        version 1.0

        BSgate(0.7853981633974483, 0) | [1, 2]
        Rgate(0.7853981633974483) | 1
        """

    program = pq.Program()

    program.loads_blackbird(blackbird_code)

    assert len(program.operations) == 2

    assert program.operations[0] == pq.B(theta=np.pi / 4, phi=0.0).on_modes(1, 2)
    assert program.operations[1] == pq.R(phi=np.pi / 4).on_modes(1)


def test_loads_blackbird_parses_operations_with_default_arguments():
    blackbird_code = \
        """name StateTeleportation
        version 1.0

        BSgate() | [1, 2]
        Rgate(0.7853981633974483) | 1
        """

    program = pq.Program()

    program.loads_blackbird(blackbird_code)

    assert len(program.operations) == 2

    assert program.operations[0] == pq.B(theta=0.0, phi=np.pi / 4).on_modes(1, 2)
    assert program.operations[1] == pq.R(phi=np.pi / 4).on_modes(1)


def test_loads_blackbird_parses_operations_with_classes_from_plugin():
    blackbird_code = \
        """name StateTeleportation
        version 1.0

        BSgate(0.7853981633974483, 0) | [1, 2]
        Rgate(0.7853981633974483) | 1
        """

    class MyBeamsplitter(pq.B):
        pass

    class Plugin:
        classes = {
            "B": MyBeamsplitter,
        }

    pq.use(Plugin)

    program = pq.Program()

    program.loads_blackbird(blackbird_code)

    assert len(program.operations) == 2

    assert program.operations[0].__class__ is MyBeamsplitter


def test_loads_blackbird_preserves_exising_operations():
    blackbird_code = \
        """name StateTeleportation
        version 1.0

        BSgate(0.7853981633974483, 0) | [1, 2]
        Rgate(0.7853981633974483) | 1
        """

    program = pq.Program()

    squeezing = pq.S(r=np.log(2), phi=np.pi / 2)

    with program:
        pq.Q() | pq.GaussianState(d=3)

        pq.Q(0) | squeezing

    program.loads_blackbird(blackbird_code)

    assert len(program.operations) == 3

    assert program.operations[0] == squeezing
    assert program.operations[1] == pq.B(theta=np.pi / 4, phi=0.0).on_modes(1, 2)
    assert program.operations[2] == pq.R(phi=np.pi / 4).on_modes(1)


def test_loads_blackbird_with_execution(gaussian_state_assets):
    blackbird_code = \
        """name StateTeleportation
        version 1.0

        BSgate(0.7853981633974483, 0) | [1, 2]
        Rgate(0.7853981633974483) | 1
        """

    program = pq.Program()

    squeezing = pq.S(r=np.log(2), phi=np.pi / 2)

    with program:
        pq.Q() | pq.GaussianState(d=3)

        pq.Q(1) | squeezing

    program.loads_blackbird(blackbird_code)

    program.execute()

    expected_state = gaussian_state_assets.load()

    assert program.state == expected_state


def test_load_blackbird_from_file_with_execution(gaussian_state_assets, tmpdir):
    blackbird_code = \
        """name StateTeleportation
        version 1.0

        BSgate(0.7853981633974483, 0) | [1, 2]
        Rgate(0.7853981633974483) | 1
        """

    blackbird_file = tmpdir.join("example-blackbird-code.xbb")

    blackbird_file.write(blackbird_code)

    program = pq.Program()

    squeezing = pq.S(r=np.log(2), phi=np.pi / 2)

    with program:
        pq.Q() | pq.GaussianState(d=3)

        pq.Q(1) | squeezing

    program.load_blackbird(blackbird_file.strpath)

    program.execute()

    expected_state = gaussian_state_assets.load()

    assert program.state == expected_state
