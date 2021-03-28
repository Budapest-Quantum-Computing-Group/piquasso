#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import piquasso as pq


def test_program_copy():
    program = pq.Program()

    program_copy = program.copy()

    assert program_copy is not program
