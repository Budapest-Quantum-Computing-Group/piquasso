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

import torch


class MockNumpy:
    """A mock-up class implementing torch versions of the
    functions used by piquasso that can be usually found
    in numpy. This class will be used by TorchConnector."""

    def __getattribute__(self, name: str, /) -> Any:
        """
        NOTE: Created to reduce boilerplate torch calls.
        """
        if hasattr(torch, name):
            return getattr(torch, name)

        return super().__getattribute__(name)

    @staticmethod
    def copy(input):
        return input.detach().clone()

    @staticmethod
    def array(input):
        # NOTE(TR): Does it make sense? Seems a little misleading,
        # but the intuition seems correct.
        return torch.Tensor(input)
