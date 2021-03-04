#
# Copyright (C) 2020 by TODO - All rights reserved.
#


class Result:
    def __init__(self, measurement, outcome):
        self.measurement = measurement
        self.outcome = outcome

    def __repr__(self):
        return f"Result of {self.measurement} with outcome '{self.outcome}'."
