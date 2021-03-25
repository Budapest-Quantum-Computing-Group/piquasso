#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import os
import sys
import random


class _Constants:
    """Module for storing constants."""

    _HBAR_DEFAULT = 2

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
