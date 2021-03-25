#
# Copyright (C) 2020 by TODO - All rights reserved.
#


class Result:
    def __init__(self, instruction, outcome):
        self.instruction = instruction
        self.outcome = outcome

    def __repr__(self):
        return f"<Result instruction={self.instruction} outcome={self.outcome}>"
