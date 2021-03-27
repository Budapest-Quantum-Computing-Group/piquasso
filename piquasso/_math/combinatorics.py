#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from itertools import chain, combinations


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
