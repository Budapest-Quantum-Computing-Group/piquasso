# %% [markdown]
# # Finding dense k-subgraphs using Gaussian Boson Sampling

# %% [markdown]
# In this tutorial, we roughly follow Ref. <cite data-footcite="graphgbs"></cite> to implement a stochastic algorithm for finding dense k-subgraphs of a graph.

# %% [markdown]
# By embedding the adjacency matrix $A$ into a Gaussian Boson Sampling <cite data-footcite="hamilton2017"></cite> (GBS) setup via the Takagi-Autonne decomposition described in Ref. <cite data-footcite="Houde_2024"></cite>, one obtains a sampler that preferentially produces outcomes corresponding to subgraphs with many perfect matchings. This property can be utilized in applications of GBS to graph problems, such as finding dense subgraphs, since graphs with many perfect matchings are typically dense. The density of a $k$-subgraph can be characterized by the ratio of the number of edges and the number of vertices, and a perfect matching is a matching (a set of pairwise non-adjacent edges) that covers every vertex of the graph. Consequently, GBS can accelerate heuristic classical algorithms that rely on subgraph sampling as a subroutine, including a stochastic algorithm for the densest $k$-subgraph problem <cite data-footcite="graphgbs"></cite>:
# given a graph $G$ and $k < |G|$, find a subgraph of $k$ vertices with the largest density!
# This problem has a natural connection to clustering problems with the goal of finding highly correlated subsets of data, having a wide range of applications.
# However, this problem is NP-hard <cite data-footcite="dense_subgraph"></cite>, and acquiring an approximation for the densest $k$-subgraph is also believed to be inefficient <cite data-footcite="bhaskara2010detectinghighlogdensities"></cite>. One alternative approach is to make use of stochastic algorithms to provide dense $k$-subgraphs.

# %% [markdown]
# We will introduce a hybrid stochastic algorithm, which uses GBS to provide guesses for dense subgraphs, originally described in Ref. <cite data-footcite="graphgbs"></cite>.
# By previous discussions, the probability of observing a given photon configuration is proportional to the number of perfect matchings in the corresponding subgraph.
# Also, intuitively, a graph with many perfect matchings is expected to contain many edges. This has been made rigorous in Ref. <cite data-footcite="aaghabali2014upperboundsnumberperfect"></cite>.
#
# We can embed the adjacency matrix in the GBS circuit, and we can obtain samples corresponding to subgraphs with a high number of perfect matchings. However, we can also exploit the obtained samples, i.e., we can provide an enhanced strategy that tweaks the samples obtained from GBS, possibly increasing the density. This algorithm can be used within an optimization algorithm searching for dense subgraphs, e.g., in a simulated annealing algorithm <cite data-footcite="simulated_annealing"></cite>. It is important to mention that from the resulting samples, we have to postselect the samples that correspond to $k$-subgraphs, i.e., samples that only contain $0$s and $1$s (called collision-free samples), and the number of $1$s is exactly $k$. In such a sample, the indices where the $1$s appear correspond to the indices of the vertices in the sampled subgraph.

# %% [markdown]
# Let us first start with the core of the algorithm, the dense $k$-subgraph sampling with GBS. This can be done by using the [Graph](../instructions/gates.rst#piquasso.instructions.gates.Graph) instruction. We also use `dask` with the `use_dask=True` configuration variable, so make sure to have it installed, or just set `use_dask=False`. We run the algorithm for a random Erdős-Rényi graph with 20 edges with $p=0.5$ probability for edge creation.

# %%
import networkx as nx
import numpy as np
import random
import piquasso as pq

from functools import partial

SEED = [12345678]

# networkx object for representing the graph
MAIN_GRAPH = nx.erdos_renyi_graph(20, p=0.5, seed=SEED[0])

k = 8  # The number of vertices in the dense subgraph
shots = 50  # Number of samples to be taken from the GBS distribution




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

    result = simulator.execute(program, shots=shots)

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
