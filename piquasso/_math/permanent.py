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


from theboss.boson_sampling_utilities.permanent_calculators.glynn_gray_permanent_calculator import (  # noqa: E501
    GlynnGrayPermanentCalculator,
)


def _permanent(matrix, rows, columns, calculator_class):
    calculator = calculator_class(matrix, rows, columns)

    return calculator.compute_permanent()


def glynn_gray_permanent(matrix, rows, columns):
    return _permanent(
        matrix, rows, columns, calculator_class=GlynnGrayPermanentCalculator
    )
