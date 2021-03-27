#
# Copyright (C) 2020 by TODO - All rights reserved.
#


class Result:
    def __init__(self, instruction, samples):
        self.instruction = instruction
        self.samples = samples

    def __repr__(self):
        return f"<Result instruction={self.instruction} samples={self.samples}>"
