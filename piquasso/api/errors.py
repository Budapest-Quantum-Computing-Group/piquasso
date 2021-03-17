#
# Copyright (C) 2020 by TODO - All rights reserved.
#


class StatePreparationError(Exception):
    """Raised when an invalid state is being prepared."""


class InvalidState(Exception):
    """Raised when an invalid state is encountered."""


class InvalidParameter(Exception):
    """Raised when an invalid parameter is specified."""
