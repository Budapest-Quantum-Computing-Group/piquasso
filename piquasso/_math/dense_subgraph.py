# This file uses source code from the files strawberryfields/apps/subgraph.py from
# https://github.com/XanaduAI/strawberryfields/blob/master/strawberryfields/apps/subgraph.py,
# Copyright 2019 Xanadu Quantum Technologies Inc. licensed under the Apache 2.0 license.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import networkx as nx
import numpy as np


def update_dict(d: dict, d_new: dict, max_count: int) -> None:
    """Updates dictionary ``d`` with subgraph tuples contained in ``d_new``.

    Subgraph tuples are a pair of values: a float specifying the subgraph density and a list of
    integers specifying the subgraph nodes. Both ``d`` and ``d_new`` are dictionaries over
    different subgraph sizes. The values of ``d`` are lists of subgraph tuples containing the top
    densest subgraphs for a given size, with maximum length ``max_count``. The values of
    ``d_new`` are candidate subgraph tuples that can be the result of resizing an input subgraph
    over a range using :func:`resize`. We want to add these candidates to the list of subgraph
    tuples in ``d`` to build up our collection of dense subgraphs.

    Args:
        d (dict[int, list[tuple[float, list[int]]]]): dictionary of subgraph sizes and
            corresponding list of subgraph tuples
        d_new (dict[int, tuple[float, list[int]]]): dictionary of subgraph sizes and corresponding
            subgraph tuples that are candidates to be added to the list
        max_count (int):  the maximum length of every subgraph tuple list

    Returns:
        None: this function modifies the dictionary ``d`` in place
    """
    for size, t in d_new.items():
        l = d.setdefault(size, [t])
        update_subgraphs_list(l, t, max_count)


def update_subgraphs_list(l: list, t: tuple, max_count: int) -> None:
    """Updates list of top subgraphs with a candidate.

    Here, the list ``l`` to be updated is a list of tuples with each tuple being a pair of
    values: a float specifying the subgraph density and a list of integers specifying the
    subgraph nodes. For example, ``l`` may be:

    ``[(0.8, [0, 5, 9, 10]), (0.5, [1, 2, 5, 6]), (0.3, [0, 4, 6, 9])]``

    We want to update ``l`` with a candidate tuple ``t``, which should be a pair specifying a
    subgraph density and corresponding subgraph nodes. For example, we might want to add:

    ``(0.4, [1, 4, 9, 10])``

    This function checks:

    - if ``t`` is already an element of ``l``, do nothing (i.e., so that ``l`` never has
      repetitions)

    - if ``len(l) < max_count``, add ``t``

    - otherwise, if the density of ``t`` exceeds the minimum density of ``l`` , add ``t`` and
      remove the element with the minimum density

    - otherwise, if the density of ``t`` equals the minimum density of ``l``, flip a coin and
      randomly swap in ``t`` with the minimum element of ``l``.

    The list ``l`` is also sorted so that its first element is the subgraph with the highest
    density.

    Args:
        l (list[tuple[float, list[int]]]): the list of subgraph tuples to be updated
        t (tuple[float, list[int]): the candidate subgraph tuple
        max_count (int): the maximum length of ``l``

    Returns:
        None: this function modifies ``l`` in place
    """
    t = (t[0], sorted(set(t[1])))

    for _d, s in l:
        if t[1] == s:
            return

    if len(l) < max_count:
        l.append(t)
        l.sort(reverse=True)
        return

    l_min = l[-1][0]

    if t[0] > l_min:
        l.append(t)
        l.sort(reverse=True)
        del l[-1]
    elif t[0] == l_min:
        if np.random.choice(2):
            del l[-1]
            l.append(t)
            l.sort(reverse=True)

    return


def resize(
        subgraph: list,
        graph: nx.Graph,
        min_size: int,
        max_size: int,
) -> dict:
    """Resize a subgraph to a range of input sizes.

    This function uses a greedy approach to iteratively add or remove nodes one at a time to an
    input subgraph to reach the range of sizes specified by ``min_size`` and ``max_size``.

    When growth is required, the algorithm examines all nodes from the remainder of the graph as
    candidates and adds the single node with the highest degree relative to the rest of the
    subgraph. This results in a graph that is one node larger, and if growth is still required,
    the algorithm performs the procedure again.

    When shrinking is required, the algorithm examines all nodes from within the subgraph as
    candidates and removes the single node with lowest degree relative to the subgraph.

    In both growth and shrink phases, there may be multiple candidate nodes with equal degree to
    add to or remove from the subgraph. The method of selecting the node is specified by the
    ``node_select`` argument, which can be either:

    - ``"uniform"`` (default): choose a node from the candidates uniformly at random;
    - A list or array: specifying the node weights of the graph, resulting in choosing the node
      from the candidates with the highest weight (when growing) and lowest weight (when shrinking),
      settling remaining ties by uniform random choice.

    **Example usage:**

    >>> s = data.Planted()
    >>> g = nx.Graph(s.adj)
    >>> s = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    >>> resize(s, g, 8, 12)
    {10: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
     11: [11, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
     12: [0, 11, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
     9: [20, 21, 22, 24, 25, 26, 27, 28, 29],
     8: [20, 21, 22, 24, 25, 26, 27, 29]}

    Args:
        subgraph (list[int]): a subgraph specified by a list of nodes
        graph (nx.Graph): the input graph
        min_size (int): minimum size for subgraph to be resized to
        max_size (int): maximum size for subgraph to be resized to
        node_select (str, list or array): method of settling ties when more than one node of
            equal degree can be added/removed. Can be ``"uniform"`` (default), or a NumPy array or
            list containing node weights.

    Returns:
        dict[int, list[int]]: a dictionary of different sizes with corresponding subgraph
    """
    nodes = graph.nodes()
    subgraph = set(subgraph)

    starting_size = len(subgraph)

    if min_size <= starting_size <= max_size:
        resized = {starting_size: sorted(subgraph)}
    else:
        resized = {}

    if max_size > starting_size:

        grow_subgraph = graph.subgraph(subgraph).copy()

        while grow_subgraph.order() < max_size:
            grow_nodes = grow_subgraph.nodes()
            complement_nodes = nodes - grow_nodes

            degrees = np.array(
                [(c, graph.subgraph(list(grow_nodes) + [c]).degree()[c]) for c in complement_nodes]
            )
            degrees_max = np.argwhere(degrees[:, 1] == degrees[:, 1].max()).flatten()

            to_add_index = np.random.choice(degrees_max)

            to_add = degrees[to_add_index][0]
            grow_subgraph.add_node(to_add)
            new_size = grow_subgraph.order()

            if min_size <= new_size <= max_size:
                resized[new_size] = sorted(grow_subgraph.nodes())

    if min_size < starting_size:

        shrink_subgraph = graph.subgraph(subgraph).copy()

        while shrink_subgraph.order() > min_size:
            degrees = np.array(shrink_subgraph.degree)
            degrees_min = np.argwhere(degrees[:, 1] == degrees[:, 1].min()).flatten()

            to_remove_index = np.random.choice(degrees_min)

            to_remove = degrees[to_remove_index][0]
            shrink_subgraph.remove_node(to_remove)

            new_size = shrink_subgraph.order()

            if min_size <= new_size <= max_size:
                resized[new_size] = sorted(shrink_subgraph.nodes())

    return resized
