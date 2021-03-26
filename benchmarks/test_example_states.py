#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np

from thewalrus.quantum import conversions


def test_example_programs_result_in_the_same_state(
    example_pq_gaussian_state,
    example_sf_gaussian_state,
):
    # NOTE: While in SF they use the xp-ordered mean and covariance by default,
    # we access it by the `xp_` prefixes.
    assert np.allclose(
        example_pq_gaussian_state.xp_mean,
        example_sf_gaussian_state.means(),
    )

    # NOTE: We use a different definition for the covariance in piquasso, that is the
    # reason for the scaling by 2.
    assert np.allclose(
        example_pq_gaussian_state.xp_cov / 2,
        example_sf_gaussian_state.cov(),
    )

    assert np.allclose(
        example_pq_gaussian_state.husimi_cov,
        conversions.Qmat(example_sf_gaussian_state.cov(), hbar=2)
    )
