#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import os
import pytest
import numpy as np

import piquasso as pq

from pathlib import Path


@pytest.fixture
def generate_symmetric_matrix():
    def func(N):
        A = np.random.rand(N, N)

        return A + A.T

    return func


@pytest.fixture
def AssetHandler(request):
    class _AssetHandler:

        def __init__(self):
            testfile = request.module.__file__
            self._asset_dir = Path(testfile).parent / "assets"

            if not os.path.exists(self._asset_dir):
                os.makedirs(self._asset_dir)

            name = request.function.__name__ + "."

            if request.cls:
                name = request.cls.__name__ + "." + name

            self._asset_id = Path(os.path.splitext(testfile)[0]).name + "." + name

        def load(self, asset: str, loader: any = None):
            loader = loader or self._load_numpy_array

            filename = self._resolve_file_for_load(asset)

            return loader(filename)

        def save(self, asset: str, obj: any):
            filename = self._resolve_file_for_save(asset)

            self._write_repr(filename, obj)

        def _resolve_file_for_load(self, asset):
            pattern = self._asset_id + asset

            files = list(Path(self._asset_dir).glob(pattern))

            assert len(files) == 1, (
                f"There should be at least one asset file named '{pattern}' "
                f"under '{self._asset_dir}'."
            )

            return files[0]

        @staticmethod
        def _load_numpy_array(filename: str):
            with open(filename, "r") as f:
                data = f.read()

            # NOTE: This is needed for the evaluation with `eval` below, where `array`'s
            # are called.
            from numpy import array  # noqa: F401

            return eval(data)

        def _resolve_file_for_save(self, asset):
            return self._asset_dir / (self._asset_id + asset)

        @staticmethod
        def _write_repr(filename, obj):
            with open(filename, "w") as outfile:
                outfile.write(repr(obj))

    return _AssetHandler


@pytest.fixture
def assets(AssetHandler):
    return AssetHandler()


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
                    "mean": obj.mean,
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
