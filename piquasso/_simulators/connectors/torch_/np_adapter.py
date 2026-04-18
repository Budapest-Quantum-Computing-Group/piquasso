#
# Copyright 2021-2026 Budapest Quantum Computing Group
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

# This module contains mock-ups used by the TorchConnector.

from typing import Any

import numpy as np
import torch


def translate_dtype(dtype):

    if dtype == np.float32:
        return torch.float32

    if dtype == np.float64:
        return torch.float64

    if dtype == np.complex64:
        return torch.complex64

    if dtype == np.complex128:
        return torch.complex128

    if dtype == np.int64:
        return torch.int64

    return dtype


class NumpyAdapter:
    """A mock-up class implementing torch versions of the
    functions used by piquasso that can be usually found
    in numpy. This class will be used by TorchConnector."""

    def __getattribute__(self, name: str, /) -> Any:
        """
        NOTE: Created to reduce boilerplate torch calls.
        """
        try:
            # Prioritize self implementations.
            return super().__getattribute__(name)
        except AttributeError:
            if hasattr(torch, name):
                return getattr(torch, name)

            raise

    @staticmethod
    def copy(input):
        return input.detach().clone()

    @property
    def int64(self):
        # Not sure if this is needed. Sometimes `fallback_np` uses `np.int64`, but `torch.int64` is not recognized.
        return np.int64

    @staticmethod
    def array(input, dtype=None):
        # NOTE(TR): Does it make sense? Seems a little misleading,
        # but the intuition seems correct.
        #
        # This method is unreasonably difficult to implement.
        if isinstance(input, torch.Tensor):
            return input

        if isinstance(input, np.ndarray):
            return torch.from_numpy(input)

        try:
            # 2D list
            return torch.stack([torch.stack(row) for row in input])
        except Exception:
            # Last resort
            return torch.tensor(input, dtype=translate_dtype(dtype))
            raise

    @staticmethod
    def transpose(input):
        return input.t()

    @staticmethod
    def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        return NumpyAdapter.array(np.isclose(a, b, rtol, atol, equal_nan))

    @staticmethod
    def mod(a, b):
        return torch.remainder(a, b)

    @staticmethod
    def matmul(a, b):
        # NOTE: torch.Tensors do not autocast during the multiplication.
        dtype: torch.dtype = torch.promote_types(
            translate_dtype(a.dtype), translate_dtype(b.dtype)
        )

        if isinstance(a, np.ndarray):
            a = torch.from_numpy(a)

        return a.to(dtype) @ b.to(dtype)

    @staticmethod
    def identity(n, dtype=torch.float32):
        return torch.eye(n, dtype=dtype)

    @staticmethod
    def astype(input, dtype):
        return input.to(dtype)

    @staticmethod
    def zeros(*args, **kwargs):
        if "dtype" in kwargs:
            kwargs["dtype"] = translate_dtype(kwargs["dtype"])

        if "shape" in kwargs:
            shape = kwargs.pop("shape")

            return torch.zeros(shape, **kwargs)
        else:
            shape = args[0]

        return torch.zeros(shape, **kwargs)
