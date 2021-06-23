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
import sys
import random


class _Constants:
    """Module for storing constants."""

    _HBAR_DEFAULT = 2

    cache_size = 32

    def __init__(self):
        self.reset_hbar()
        self.seed()

    def reset_hbar(self):
        self.HBAR = self._HBAR_DEFAULT

    def seed(self, sequence=None):
        self._SEED = sequence or int.from_bytes(os.urandom(8), byteorder="big")
        random.seed(self._SEED)

    def get_seed(self):
        return self._SEED

    def __repr__(self):
        return f"<Constants HBAR={self.HBAR} SEED={self.get_seed()}>"


sys.modules[__name__] = _Constants()
