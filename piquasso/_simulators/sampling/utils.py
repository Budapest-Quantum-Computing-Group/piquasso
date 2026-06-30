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

from piquasso._math.combinatorics import partitions, partitions_bounded_k

"""
This is a contribution from `theboss`, see https://github.com/Tomev-CTP/theboss.

The original code has been re-implemented and adapted to Piquasso.
"""

__author__ = "Tomasz Rybotycki"


def calculate_state_vector(
    interferometer,
    initial_state,
    postselect_data,
    config,
    connector,
):
    """Calculate the state vector on the particle subspace defined by ``initial_state``.

    This implementation follows Algorithm 1 (``SLOS_full``) from
    `Strong Simulation of Linear Optical Processes`.

    The algorithm is modified by pruning to account for postselection on certain modes.
    """

    np = connector.np
    fallback_np = connector.fallback_np

    is_postselected = len(postselect_data[0]) > 0

    d = len(initial_state)
    n = int(fallback_np.sum(initial_state))

    if is_postselected:
        """
        The idea is the following:

        Let's say that we want to do a full state vector simulation on 3 modes for 4
        photons, but we only want the components in the state vector for which the
        particle number in the last two modes are, e.g., (1, 1). So, for us, all the
        interesting Fock basis states are |2, 0, 1, 1>, |1, 1, 1, 1>, |0, 2, 1, 1>. In
        the k_limit=0 case, the `partitions_bounded_k` function calculates these.

        However, let's say that you're calculating the state vector with SLOS, meaning
        that you are calculating the state vector particle-by-particle. Then of course,
        you cannot have a transition, e.g., |2, 1, 0, 0> -> |x, y, 1, 1> (where x+y=2),
        so you would like to disregard |2, 1, 0, 0>, but for example |1, 1, 0, 1> would
        be fine. More specifically, the condition is to let the difference on the
        postselected modes be up to some number k_limit, where we will set k_limit
        according to the number of particles we still need to add to the state vector
        (because they could end up in the postselected modes still).
        """
        postselect_modes, postselect_photons = postselect_data
        bases = [
            partitions_bounded_k(
                boxes=d,
                particles=k,
                constrained_boxes=postselect_modes,
                max_per_box=postselect_photons,
                k_limit=(n - k),
            )
            for k in range(n + 1)
        ]
    else:
        bases = [partitions(boxes=d, particles=k) for k in range(n + 1)]

    index_maps = [{tuple(basis[i]): i for i in range(len(basis))} for basis in bases]

    sigma = fallback_np.zeros(d, dtype=int)
    schedule_sigma = []
    schedule_mode = []
    for p in range(d):
        for _ in range(initial_state[p]):
            schedule_sigma.append(sigma.copy())
            schedule_mode.append(p)
            sigma[p] += 1

    UF_curr = np.zeros(len(bases[0]), dtype=config.complex_dtype)
    UF_curr = connector.assign(UF_curr, 0, 1.0)

    for k in range(n):
        sigma_k = schedule_sigma[k]
        p = schedule_mode[k]
        index_map = index_maps[k + 1]
        UF_next = np.zeros(len(bases[k + 1]), dtype=config.complex_dtype)

        for idx_t, t in enumerate(bases[k]):
            A = UF_curr[idx_t]

            for i in range(d):
                t_i = t[i]
                new_t = t.copy()
                new_t[i] += 1
                tuple_new_t = tuple(new_t)
                if tuple_new_t not in index_map:
                    continue

                index_new = index_map[tuple_new_t]

                factor = fallback_np.sqrt((t_i + 1) / (sigma_k[p] + 1))

                UF_next = connector.assign(
                    UF_next,
                    index_new,
                    UF_next[index_new] + factor * interferometer[i, p] * A,
                )

        UF_curr = UF_next

    return UF_curr


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
