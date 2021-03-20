#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest

import numpy as np

import piquasso as pq


@pytest.fixture
def gaussian_state_assets(AssetHandler):

    class GaussianStateAssetHandler(AssetHandler):
        def __init__(self):
            super().__init__()

            self._asset = "expected_state"

        def load(self):
            filename = self._resolve_file_for_load(asset=self._asset)

            return self._load_gaussian_state(filename)

        def save(self, obj: any):
            state_map = {
                "class": "GaussianState",
                "properties": {
                    "mean": obj.mu,
                    "cov": obj.cov,
                }
            }

            filename = self._resolve_file_for_save(self._asset)

            self._write_repr(filename, state_map)

        @staticmethod
        def _load_gaussian_state(filename):
            """
            Note:
                Do not use the `eval` approach ever in production code, it's insecure!
            """
            with open(filename, "r") as f:
                data = f.read()

            # NOTE: This is needed for the evaluation with `eval` below, where `array`'s
            # are called.
            from numpy import array  # noqa: F401

            properties = eval(data)["properties"]

            d = len(properties["mean"]) // 2

            state = pq.GaussianState(d=d)

            state.mean = properties["mean"]
            state.cov = properties["cov"]

            return state

    return GaussianStateAssetHandler()


@pytest.fixture
def program():
    with pq.Program() as initialization:
        pq.Q() | pq.GaussianState(d=3)

        pq.Q(0) | pq.D(alpha=1)
        pq.Q(1) | pq.D(alpha=1j)
        pq.Q(2) | pq.D(alpha=np.exp(1j * np.pi/4))

        pq.Q(0) | pq.S(np.log(2), theta=np.pi / 2)
        pq.Q(1) | pq.S(np.log(1), theta=np.pi / 4)
        pq.Q(2) | pq.S(np.log(2), theta=np.pi / 2)

    initialization.execute()
    initialization.state.validate()

    return pq.Program(state=initialization.state)
