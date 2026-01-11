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

import numpy as np
import numba as nb


@nb.njit(cache=True)
def arr_comb(n, k):
    n = np.where((n < 0) | (n < k), 0, n)
    prod = np.ones(n.shape, dtype=np.int64)

    for i in range(k):
        prod *= n - i
        prod = prod // (i + 1)

    return prod


@nb.njit(cache=True)
def comb(n, k):
    if n < 0 or k < 0 or n < k:
        return 0

    prod = 1

    k = min(k, n - k)

    for i in range(k):
        prod *= n - i
        prod //= i + 1

    return prod


@nb.njit(cache=True)
def sort_and_get_parity(array):
    n = len(array)
    parity = 1
    for n in range(n - 1, 0, -1):
        for i in range(n):
            if array[i] > array[i + 1]:
                array[i], array[i + 1] = array[i + 1], array[i]
                parity *= -1

    return array, parity


@nb.njit(cache=True)
def partitions(boxes, particles, out=None):
    r"""
    Returns all the possible ways to put a specified number of particles in a specified
    number of boxes in anti-lexicographic order.

    Args:
        boxes: Number of boxes.
        particles: Number of particles.
        out: Optional output array.
    """

    positions = particles + boxes - 1

    if positions == -1 or boxes == 0:
        return np.empty((1, 0), dtype=np.int32)

    size = comb(positions, boxes - 1)

    if out is None:
        result = np.empty((size, boxes), dtype=np.int32)
    else:
        result = out

    separators = np.arange(boxes - 1, dtype=np.int32)
    index = size - 1

    while True:
        prev = -1
        for i in range(boxes - 1):
            result[index, i] = separators[i] - prev - 1
            prev = separators[i]

        result[index, boxes - 1] = positions - prev - 1
        index -= 1

        if index < 0:
            break

        i = boxes - 2
        while separators[i] == positions - (boxes - 1 - i):
            i -= 1

        separators[i] += 1
        for j in range(i + 1, boxes - 1):
            separators[j] = separators[j - 1] + 1

    return result


def partitions_bounded_k(boxes, particles, constrained_boxes, max_per_box, k_limit):
    """
    Returns all weak compositions x of 'particles' into 'boxes' such that:
      - for each constrained box m = constrained_boxes[i]:
            x[m] <= max_per_box[i]
      - sum over constrained boxes (max_per_box[i] - x[constrained_boxes[i]]) <= k_limit

    Result is in the same ordering as `partitions`
    (e.g. for boxes=3, particles=2:
       [2, 0, 0],
       [1, 1, 0],
       [1, 0, 1],
       [0, 2, 0],
       [0, 1, 1],
       [0, 0, 2])
    """
    constrained_boxes = np.asarray(constrained_boxes, dtype=np.int32)
    max_per_box = np.asarray(max_per_box, dtype=np.int32)

    bounds, constrained, target = _build_bounds_and_targets(
        boxes, particles, constrained_boxes, max_per_box
    )

    # First pass: count
    count = _count_partitions_bounded_k_recursive(
        0, boxes, particles, bounds, constrained, target, 0, k_limit
    )

    out = np.empty((count, boxes), dtype=np.int32)
    current = np.zeros(boxes, dtype=np.int32)

    # Second pass: fill in descending lex order
    _ = _fill_partitions_bounded_k_recursive(
        0, boxes, particles, bounds, constrained, target, 0, k_limit, current, out, 0
    )

    return out


@nb.njit(cache=True)
def _build_bounds_and_targets(boxes, particles, modes, max_per_mode):
    """
    Build:
      - bounds[b]: upper bound for box b
      - constrained[b]: True if box b is constrained
      - target[b]: the 'max_per_mode' for constrained boxes, 0 otherwise
    """
    bounds = np.empty(boxes, dtype=np.int32)
    constrained = np.zeros(boxes, dtype=np.bool_)
    target = np.zeros(boxes, dtype=np.int32)

    # default: no constraint except the trivial particles bound
    for b in range(boxes):
        bounds[b] = particles

    # apply explicit per-mode constraints
    for k in range(modes.shape[0]):
        b = modes[k]
        constrained[b] = True
        target[b] = max_per_mode[k]
        # enforce upper bound from max_per_mode
        if max_per_mode[k] < bounds[b]:
            bounds[b] = max_per_mode[k]

    return bounds, constrained, target


@nb.njit(cache=True)
def _count_partitions_bounded_k_recursive(
    box_idx,
    boxes,
    remaining,
    bounds,
    constrained,
    target,
    diff_so_far,
    k_limit,
):
    """
    Count compositions with:
      - 0 <= x[b] <= bounds[b]
      - sum_b constrained (target[b] - x[b]) <= k_limit
    """
    if remaining < 0:
        return 0

    if box_idx == boxes - 1:
        # last box takes 'remaining'
        if remaining > bounds[box_idx]:
            return 0

        # update diff
        if constrained[box_idx]:
            diff_final = diff_so_far + (target[box_idx] - remaining)
        else:
            diff_final = diff_so_far

        if diff_final <= k_limit:
            return 1
        else:
            return 0

    total = 0
    max_here = bounds[box_idx]
    if max_here > remaining:
        max_here = remaining

    # Descending order: max_here, ..., 0
    for val in range(max_here, -1, -1):
        new_diff = diff_so_far
        if constrained[box_idx]:
            new_diff += target[box_idx] - val

        # prune: diff cannot decrease later
        if new_diff > k_limit:
            continue

        total += _count_partitions_bounded_k_recursive(
            box_idx + 1,
            boxes,
            remaining - val,
            bounds,
            constrained,
            target,
            new_diff,
            k_limit,
        )

    return total


@nb.njit(cache=True)
def _fill_partitions_bounded_k_recursive(
    box_idx,
    boxes,
    remaining,
    bounds,
    constrained,
    target,
    diff_so_far,
    k_limit,
    current,
    out,
    write_idx,
):
    """
    Fill result array with all compositions satisfying the constraints,
    in descending lexicographic order w.r.t. x[0], x[1], ...
    """
    if remaining < 0:
        return write_idx

    if box_idx == boxes - 1:
        if remaining > bounds[box_idx]:
            return write_idx

        if constrained[box_idx]:
            diff_final = diff_so_far + (target[box_idx] - remaining)
        else:
            diff_final = diff_so_far

        if diff_final <= k_limit:
            current[box_idx] = remaining
            # copy to out[write_idx]
            for b in range(boxes):
                out[write_idx, b] = current[b]
            return write_idx + 1
        else:
            return write_idx

    max_here = bounds[box_idx]
    if max_here > remaining:
        max_here = remaining

    # Descending over this coordinate gives descending lex over full vector
    for val in range(max_here, -1, -1):
        new_diff = diff_so_far
        if constrained[box_idx]:
            new_diff += target[box_idx] - val

        if new_diff > k_limit:
            continue

        current[box_idx] = val
        write_idx = _fill_partitions_bounded_k_recursive(
            box_idx + 1,
            boxes,
            remaining - val,
            bounds,
            constrained,
            target,
            new_diff,
            k_limit,
            current,
            out,
            write_idx,
        )

    return write_idx
