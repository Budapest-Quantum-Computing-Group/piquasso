# This file uses source code from the files strawberryfields/apps/clique.py from
# https://github.com/XanaduAI/strawberryfields/blob/master/strawberryfields/apps/clique.py,
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


def grow(clique: list, graph: nx.Graph) -> list:
    """Iteratively adds new nodes to the input clique to generate a larger clique.

    Each iteration involves calculating the set :math:`C_0` (provided by the function
    :func:`c_0`) with respect to the current clique. This set represents the nodes in the rest of
    the graph that are connected to all of the nodes in the current clique. Therefore, adding any of
    the nodes in :math:`C_0` will create a larger clique. This function proceeds by repeatedly
    evaluating :math:`C_0` and selecting and adding a node from this set to add to the current
    clique. Growth is continued until :math:`C_0` becomes empty.

    Whenever there are multiple nodes within :math:`C_0`, one must choose which node to add to
    the growing clique. This function allows a method of choosing nodes to be set with the
    ``node_select`` argument, which can be any of the following:

    - ``"uniform"`` (default): choose a node from the candidates uniformly at random;
    - ``"degree"``: choose the node from the candidates with the greatest degree, settling ties
      by uniform random choice;
    - A list or array: specifying the node weights of the graph, resulting in choosing the node
      from the candidates with the greatest weight, settling ties by uniform random choice.

    **Example usage:**

    >>> graph = nx.complete_graph(10)
    >>> clique = [0, 1, 2, 3, 4]
    >>> grow(clique, graph)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    Args:
        clique (list[int]): a subgraph specified by a list of nodes; the subgraph must be a clique
        graph (nx.Graph): the input graph
        node_select (str, list or array): method of selecting nodes from :math:`C_0` during
            growth. Can be ``"uniform"`` (default), ``"degree"``, or a NumPy array or list.

    Returns:
        list[int]: a new clique subgraph of equal or larger size than the input
    """

    if not set(clique).issubset(graph.nodes):
        raise ValueError("Input is not a valid subgraph")

    if not is_clique(graph.subgraph(clique)):
        raise ValueError("Input subgraph is not a clique")

    clique = set(clique)
    _c_0 = sorted(c_0(clique, graph))

    while _c_0:
        clique.add(np.random.choice(_c_0))
        _c_0 = sorted(c_0(clique, graph))

    return sorted(clique)


def swap(clique: list, graph: nx.Graph) -> list:
    """If possible, generates a new clique by swapping a node in the input clique with a node
    outside the clique.

    Proceeds by calculating the set :math:`C_1` of nodes in the rest of the graph that are
    connected to all but one of the nodes in the clique. If this set is not empty, this function
    randomly picks a node and swaps it with the corresponding node in the clique that is not
    connected to it. The set :math:`C_1` and corresponding nodes in the clique are provided by the
    :func:`c_1` function.

    Whenever there are multiple nodes within :math:`C_1`, one must choose which node to add to
    the growing clique. This function allows a method of choosing nodes to be set with the
    ``node_select`` argument, which can be any of the following:

    - ``"uniform"`` (default): choose a node from the candidates uniformly at random;
    - ``"degree"``: choose the node from the candidates with the greatest degree, settling ties
      by uniform random choice;
    - A list or array: specifying the node weights of the graph, resulting in choosing the node
      from the candidates with the greatest weight, settling ties by uniform random choice.

    **Example usage:**

    >>> graph = nx.wheel_graph(5)
    >>> graph.remove_edge(0, 4)
    >>> clique = [0, 1, 2]
    >>> swap(clique, graph)
    [0, 2, 3]

    Args:
        clique (list[int]): a subgraph specified by a list of nodes; the subgraph must be a clique
        graph (nx.Graph): the input graph
        node_select (str, list or array): method of selecting incoming nodes from :math:`C_1`
            during swapping. Can be ``"uniform"`` (default), ``"degree"``, or a NumPy array or list.

    Returns:
        list[int]: a new clique subgraph of equal size as the input
    """

    if not set(clique).issubset(graph.nodes):
        raise ValueError("Input is not a valid subgraph")

    if not is_clique(graph.subgraph(clique)):
        raise ValueError("Input subgraph is not a clique")

    clique = set(clique)
    _c_1 = c_1(clique, graph)

    if _c_1:
        swap_index = np.random.choice(len(_c_1))
        swap_nodes = _c_1[swap_index]

        clique.remove(swap_nodes[0])
        clique.add(swap_nodes[1])

    return sorted(clique)


def is_clique(graph: nx.Graph) -> bool:
    """Determines if the input graph is a clique. A clique of :math:`n` nodes has exactly :math:`n(
    n-1)/2` edges.

    **Example usage:**

    >>> graph = nx.complete_graph(10)
    >>> is_clique(graph)
    True

    Args:
        graph (nx.Graph): the input graph

    Returns:
        bool: ``True`` if input graph is a clique and ``False`` otherwise
    """
    edges = graph.edges
    nodes = graph.order()

    return len(edges) == nodes * (nodes - 1) / 2


def c_0(clique: list, graph: nx.Graph):
    """Generates the set :math:`C_0` of nodes that are connected to all nodes in the input
    clique subgraph.

    The set :math:`C_0` is defined in :cite:`pullan2006phased` and is used to determine nodes
    that can be added to the current clique to grow it into a larger one.

    **Example usage:**

    >>> graph = nx.complete_graph(10)
    >>> clique = [0, 1, 2, 3, 4]
    >>> c_0(clique, graph)
    [5, 6, 7, 8, 9]

    Args:
        clique (list[int]): a subgraph specified by a list of nodes; the subgraph must be a clique
        graph (nx.Graph): the input graph

    Returns:
        list[int]: a list containing the :math:`C_0` nodes for the clique

    """
    if not is_clique(graph.subgraph(clique)):
        raise ValueError("Input subgraph is not a clique")

    clique = set(clique)
    c_0_nodes = []
    non_clique_nodes = set(graph.nodes) - clique

    for i in non_clique_nodes:
        if clique.issubset(graph.neighbors(i)):
            c_0_nodes.append(i)

    return c_0_nodes


def c_1(clique: list, graph: nx.Graph):
    """Generates the set :math:`C_1` of nodes that are connected to all but one of the nodes in
    the input clique subgraph.

    The set :math:`C_1` is defined in :cite:`pullan2006phased` and is used to determine outside
    nodes that can be swapped with clique nodes to create a new clique.

    **Example usage:**

    >>> graph = nx.wheel_graph(5)
    >>> clique = [0, 1, 2]
    >>> c_1(clique, graph)
    [(1, 3), (2, 4)]

    Args:
        clique (list[int]): a subgraph specified by a list of nodes; the subgraph must be a clique
        graph (nx.Graph): the input graph

    Returns:
       list[tuple[int]]: A list of tuples. The first node in the tuple is the node in the clique
       and the second node is the outside node it can be swapped with.
    """
    if not is_clique(graph.subgraph(clique)):
        raise ValueError("Input subgraph is not a clique")

    clique = set(clique)
    c_1_nodes = []
    non_clique_nodes = set(graph.nodes) - clique

    for i in non_clique_nodes:
        neighbors_in_subgraph = clique.intersection(graph.neighbors(i))

        if len(neighbors_in_subgraph) == len(clique) - 1:
            to_swap = clique - neighbors_in_subgraph
            (i_clique,) = to_swap
            c_1_nodes.append((i_clique, i))

    return c_1_nodes
