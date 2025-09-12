# %%
import networkx as nx
import numpy as np
import random
import piquasso as pq

from functools import partial


# networkx object for representing the graph
MAIN_GRAPH = nx.erdos_renyi_graph(20, p=0.5)

k = 8  # The number of vertices in the dense subgraph
shots = 50  # Number of samples to be taken from the GBS distribution


SEED = [1234]


def is_collision_free_length_k(nodes, k):
    return len(nodes) == len(np.unique(nodes)) and len(nodes) == k


def postselect(subgraph_nodes, k):
    return list(filter(partial(is_collision_free_length_k, k=k), subgraph_nodes))


def generate_dense_subgraph(graph, k):
    """Generates a dense k-subgraph using the GBS distribution.

    Args:
        graph (networkx.Graph): The graph to be embedded into the GBS circuit.
        k (int): Number of vertices in the sampled subgraph.

    Returns:
        networkx.Graph: A dense k-subgraph.
    """
    adjacency_matrix = nx.adjacency_matrix(graph).toarray()

    with pq.Program() as program:
        pq.Q() | pq.Vacuum() | pq.Graph(
            adjacency_matrix, mean_photon_number=k / len(graph)
        ) | pq.ParticleNumberMeasurement()

    simulator = pq.GaussianSimulator(
        d=len(graph), config=pq.Config(use_dask=True, seed_sequence=SEED[0])
    )
    SEED[0] += 1

    try:
        result = simulator.execute(program, shots=shots)
    except:
        print("Adjacency matrix:", adjacency_matrix)
        raise

    dense_subgraph_nodes = result.to_subgraph_nodes()

    filtered_dense_subgraph_nodes = postselect(dense_subgraph_nodes, k)

    if len(filtered_dense_subgraph_nodes) == 0:
        # No dense subgraph is found, call again!
        return generate_dense_subgraph(graph, k)

    randomly_chosen_nodes = random.choice(filtered_dense_subgraph_nodes)

    return graph.subgraph(np.array(graph.nodes)[randomly_chosen_nodes])


# %% [markdown]
# We can use this function to get a $k$-subgraph corresponding to a high hafnian value, implying high density:

# %%
subgraph = generate_dense_subgraph(MAIN_GRAPH, k)

print("Subgraph nodes:", subgraph.nodes)
print("Subgraph density:", nx.density(subgraph))

# %% [markdown]
# Now, we aim to tweak this subgraph, using the GBS-Tweak algorithm described in Ref. <cite data-footcite="graphgbs"></cite>:


# %%
def try_generate_tweaked_subgraph(subgraph, l):
    """
    Tries to tweak the subgraph by using the GBS distribution.

    Note: the resulting subgraph might be smaller than the original.

    Args:
        subgraph (networkx.Graph): The initial subgraph.
        l (int): Number of vertices to keep.

    Returns:
        networkx.Graph: The tweaked subgraph.
    """
    k = len(subgraph)

    R = generate_dense_subgraph(subgraph, l)  # l-subgraph of the original k-subgraph
    T = generate_dense_subgraph(MAIN_GRAPH, k - l)  # k-l-subgraph of the original graph

    # Adding m extra nodes randomly
    m = np.random.randint(0, k - l)

    # Add m nodes to R randomly from the original subgraph
    subgraph_minus_R = subgraph.copy()
    subgraph_minus_R.remove_nodes_from(R)
    extra_nodes = np.random.choice(subgraph_minus_R.nodes, size=m, replace=False)
    R_with_extra_nodes = R.copy()
    R_with_extra_nodes.add_nodes_from(extra_nodes)

    # Remove m nodes randomly from T
    reject_nodes = np.random.choice(T.nodes, size=m, replace=False)
    T_with_rejected_nodes = T.copy()
    T_with_rejected_nodes.remove_nodes_from(reject_nodes)

    nodes_to_keep = R_with_extra_nodes.nodes
    nodes_to_add = T_with_rejected_nodes.nodes

    tweaked_subgraph_nodes = np.concatenate([nodes_to_keep, nodes_to_add])

    return MAIN_GRAPH.subgraph(tweaked_subgraph_nodes)


def gbs_tweak(subgraph, l):
    """An implementation for the GBS-Tweak algorithm.

    Args:
        subgraph (networkx.Graph): The subgraph to be tweaked.
        l (int): The number of vertices to keep.

    Returns:
        networkx.Graph: The tweaked subgraph.
    """

    tweaked_subgraph = try_generate_tweaked_subgraph(subgraph, l)
    while len(tweaked_subgraph) != len(subgraph):
        # Try generating another if the resulting subgraph is smaller than the original.
        tweaked_subgraph = try_generate_tweaked_subgraph(subgraph, l)

    return tweaked_subgraph


# %% [markdown]
# A parameter of this algorithm is $l$, the number of vertices to be kept from the original subgraph. After fixing this, we can generate a new subgraph:

# %%
l = 4  # Number of vertices to be kept in the GBS-tweak algorithm

new_subgraph = gbs_tweak(subgraph, l)

print("Original subgraph:", subgraph)
print("New subgraph:", new_subgraph)

# %% [markdown]
# Now, we can implement a simple simulated annealing optimization for finding $k$-dense subgraphs.

# %%
initial_temperature = 100  # Initial temperature for simulated annealing
alpha = 0.85  # Parameter for geometric cooling
ITER = 100  # Number of iterations

subgraph = generate_dense_subgraph(MAIN_GRAPH, k)

best_subgraph_so_far = subgraph
best_density_so_far = nx.density(subgraph)

T = initial_temperature

for i in range(ITER):
    density = nx.density(subgraph)
    print(f"{i}. density={density}")
    new_subgraph = gbs_tweak(subgraph, l)  # Tweak the previous subgraph
    new_density = nx.density(new_subgraph)

    if new_density > density:
        # If the new density is higher, accept the new subgraph
        subgraph = new_subgraph
    else:
        # Otherwise, calculate an temperature-dependent acceptance probability
        probability = np.exp((new_density - density) / T)

        if np.random.rand() < probability:
            # Accept anyway with some probability
            subgraph = new_subgraph

    if new_density > best_density_so_far:
        # If the new density is higher than any density observed so far, save this
        # subgraph!
        best_subgraph_so_far = new_subgraph
        best_density_so_far = new_density

    T *= alpha  # Decrease temperature to decrease acceptance probability

print("Solution:", best_subgraph_so_far)
print("Density:", best_density_so_far)
