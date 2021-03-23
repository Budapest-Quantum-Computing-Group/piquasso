#
# Copyright (C) 2020 by TODO - All rights reserved.
#


class Result:
    def __init__(self, operation, outcome):
        self.operation = operation
        self.outcome = outcome

    def __repr__(self):
        return f"<Result operation={self.operation} outcome={self.outcome}>"
