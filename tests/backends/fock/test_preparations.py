#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np

import piquasso as pq


def test_from_pure_preserves_fock_probabilities():
    with pq.Program() as pure_state_preparation:
        pq.Q() | pq.PureFockState(d=2, cutoff=2)

        pq.Q(1) | pq.StateVector(1)

    pure_state_preparation.execute()

    beamsplitter = pq.B(theta=np.pi / 4, phi=np.pi / 3)

    with pq.Program() as pure_state_program:
        pq.Q() | pure_state_preparation.state

        pq.Q(0, 1) | beamsplitter

    with pq.Program() as mixed_state_program:
        pq.Q() | pq.FockState.from_pure(pure_state_preparation.state)

        pq.Q(0, 1) | beamsplitter

    pure_state_program.execute()
    mixed_state_program.execute()

    pure_state_fock_probabilities = pure_state_program.state.fock_probabilities
    mixed_state_fock_probabilities = mixed_state_program.state.fock_probabilities

    assert np.allclose(mixed_state_fock_probabilities, pure_state_fock_probabilities)