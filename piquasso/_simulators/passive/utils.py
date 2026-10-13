#
# Copyright 2021-2026 Budapest Quantum Computing Group
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

from scipy.special import factorial


def calculate_inner_product(interferometer, input, output, connector):
    np = connector.np
    fallback_np = connector.fallback_np
    permanent = connector.permanent(interferometer, cols=input, rows=output)

    return permanent / np.sqrt(
        fallback_np.prod(factorial(output)) * fallback_np.prod(factorial(input))
    )


def calculate_lossy_density_matrix_element(
    input_left,
    input_right,
    output_left,
    output_right,
    loss_channel_matrix,
    connector,
):
    """
    Calculates the element of the lossy density matrix for the given input and output
    states.

    Source: https://arxiv.org/abs/2412.17742
    """
    # TODO: There might be a more efficient way to compute this, especially in certain
    # cases, e.g., when the input and output states are the same, or for uniform losses.
    fallback_np = connector.fallback_np

    input_left = fallback_np.asarray(input_left, dtype=int)
    input_right = fallback_np.asarray(input_right, dtype=int)
    output_left = fallback_np.asarray(output_left, dtype=int)
    output_right = fallback_np.asarray(output_right, dtype=int)

    lost_left = int(fallback_np.sum(input_left) - fallback_np.sum(output_left))
    lost_right = int(fallback_np.sum(input_right) - fallback_np.sum(output_right))

    if lost_left < 0 or lost_right < 0:
        return 0.0

    if lost_left != lost_right:
        return 0.0

    repeated_occupation_number = fallback_np.concatenate(
        [
            input_left,
            output_left,
            input_right,
            output_right,
        ]
    )

    # NOTE: This can also be a hafnian, but that is not yet supported with JaxConnector
    hafnian = connector.loop_hafnian(
        loss_channel_matrix,
        fallback_np.zeros(loss_channel_matrix.shape[0], dtype=complex),
        repeated_occupation_number,
    )

    denominator = fallback_np.sqrt(
        fallback_np.prod(factorial(input_left))
        * fallback_np.prod(factorial(input_right))
        * fallback_np.prod(factorial(output_left))
        * fallback_np.prod(factorial(output_right))
    )

    return hafnian / denominator
