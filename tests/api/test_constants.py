

import piquasso as pq


def test_HBAR_setting():
    pq.constants.HBAR = 42

    assert pq.constants.HBAR == 42

    pq.constants.reset_hbar()  # Teardown


def test_SEED_is_set_initially():
    assert pq.constants.get_seed()


def test_SEED_setting():
    pq.constants.seed(123)

    assert pq.constants.get_seed() == 123

    pq.constants.seed()  # Teardown
