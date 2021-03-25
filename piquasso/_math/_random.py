#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import random


def choose_from_cumulated_probabilities(cumulated_probabilities):
    guess = random.uniform(0, cumulated_probabilities[-1])

    for first, second in zip(
        cumulated_probabilities, cumulated_probabilities[1:]
    ):
        if first < guess <= second:
            return cumulated_probabilities.index(first)
