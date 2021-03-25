#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np


def test_example_programs_result_in_the_same_state(
    example_gaussian_pq_program,
    example_gaussian_sf_program_and_engine,
):
    example_gaussian_pq_program.execute()
    piquasso_state = example_gaussian_pq_program.state

    example_gaussian_sf_program, engine = example_gaussian_sf_program_and_engine
    results = engine.run(example_gaussian_sf_program)
    strawberry_state = results.state

    # NOTE: While in SF they use the xp-ordered mean and covariance by default,
    # we access it by the `xp_` prefixes.
    assert np.allclose(
        piquasso_state.xp_mean,
        strawberry_state.means(),
    )

    # NOTE: We use a different definition for the covariance in piquasso, that is the
    # reason for the scaling by 2.
    assert np.allclose(
        piquasso_state.xp_cov / 2,
        strawberry_state.cov(),
    )
