#
# Copyright 2021 Budapest Quantum Computing Group
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

import os
import random
from typing import Any

import numpy as np

"""Module for storing constants."""

_HBAR_DEFAULT = 2.0

cache_size = 32

_SEED = int.from_bytes(os.urandom(8), byteorder="big")

HBAR = _HBAR_DEFAULT

RNG = np.random.default_rng(_SEED)

use_torontonian = False


def reset_hbar() -> None:
    global HBAR
    HBAR = _HBAR_DEFAULT


def seed(sequence: Any = None) -> None:
    global _SEED
    global RNG
    _SEED = sequence or int.from_bytes(os.urandom(8), byteorder="big")
    RNG = np.random.default_rng(_SEED)
    random.seed(_SEED)


def get_seed() -> Any:
    global _SEED
    return _SEED
