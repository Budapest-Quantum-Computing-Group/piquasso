import numpy as np
import scipy
import numba as nb
from piquasso._math.combinatorics import comb
from piquasso._math.combinatorics import sort_and_get_parity
from piquasso.fermionic.gaussian._misc import _get_fs_fdags, tensor_product
import piquasso as pq

connector = pq.NumpyConnector()


@nb.njit(cache=True)
def next_index(first_quantized, d):
    l = len(first_quantized)

    for i in range(l):
        if first_quantized[l - i - 1] < d - i - 1:
            first_quantized[l - i - 1] += 1
            for k in range(l - i, l):
                first_quantized[k] = first_quantized[l - i - 1] + 1

            return first_quantized

    new_majorana_string = np.empty(l + 1, dtype=first_quantized.dtype)

    for i in range(len(new_majorana_string)):
        new_majorana_string[i] = i

    return new_majorana_string


def _to_occupation_number_picture(tensor_index, N):
    ret = np.zeros(shape=8 * N, dtype=tensor_index.dtype)
    for i in range(len(tensor_index)):
        ret[tensor_index[i]] = 1

    return ret


def get_vector_lengths(N, l):
    lengths = []
    for l1 in range(2*l+1):
        lengths += [comb(8*N, l1)]
    return lengths


def get_fermion_sampling_initial_tensor(N: int, l: int):
    tensor = np.empty(shape=(8 * N,) * (2 * l), dtype=complex)
    size = (8 * N) ** (2 * l)
    psi_tensor = get_psi()
    for idx in range(size):
        tensor_index = np.empty(shape=2 * l, dtype=int)
        for i in range(2 * l):
            tensor_index[2 * l - i - 1] = idx % (8 * N)
            idx = idx // (8 * N)

        sorted_tensor_index, prefactor = reduce_and_sort_majoranas(tensor_index)

        occ_tensor_index = _to_occupation_number_picture(sorted_tensor_index, N)

        sliced = np.reshape(occ_tensor_index, (N, 8))

        prod = 1.0

        for i in range(N):
            prod *= psi_tensor[tuple(sliced[i])]

        tensor[tuple(tensor_index)] = prefactor * prod

    return tensor


def get_fermion_sampling_initial_vector(N: int, l: int):
    sub_vec_lengths = get_vector_lengths(N, l)
    size = np.sum(sub_vec_lengths)
    initial_vector = np.empty(shape=size, dtype=complex)
    psi_tensor = get_psi()
    # initialize
    current_index = 0
    majorannas = np.array([], dtype=int)
    # do the empty one by hand
    initial_vector[current_index] = 1
    current_index += 1
    majorannas = next_index(majorannas, 8*N)
    for i in range(size-1):
        occ_tensor_index = _to_occupation_number_picture(majorannas, N)
        sliced = np.reshape(occ_tensor_index, (N, 8))

        prod = 1.0

        for i in range(N):
            prod *= psi_tensor[tuple(sliced[i])]

        initial_vector[current_index] = prod
        current_index += 1
        majorannas = next_index(majorannas, 8*N)

    return initial_vector


def get_psi():
    tensor = np.empty(shape=(2,) * (2 * 4), dtype=complex)
    size = 2**8
    for idx in range(size):
        tensor_index = np.empty(shape=2 * 4, dtype=int)
        for i in range(2 * 4):
            tensor_index[2 * 4 - i - 1] = idx % (2 * 1)
            idx = idx // (2 * 1)

        first_quantized_index = np.arange(2 * 4)[tensor_index == 1]
        extended = np.empty(
            len(first_quantized_index) + 4, dtype=first_quantized_index.dtype
        )
        extended[2:-2] = first_quantized_index

        sum = 0.0
        for i in range(2):
            extended[-2:] = [4 * i, 4 * i + 2]
            for j in range(2):
                extended[:2] = [4 * j + 2, 4 * j]
                prefactor = get_prefactor(extended)
                sum += prefactor

        sum /= 2

        tensor[tuple(tensor_index)] = sum

    return tensor


def test_psi():
    psi = get_psi()
    # breakpoint()
    # empty
    assert np.isclose(psi[0, 0, 0, 0, 0, 0, 0, 0], 1)
    # two majorannas not in pair
    assert np.isclose(psi[0, 1, 1, 0, 0, 0, 0, 0], 0)
    assert np.isclose(psi[0, 0, 0, 1, 0, 0, 1, 0], 0)
    assert np.isclose(psi[0, 0, 1, 0, 0, 0, 1, 0], 0)
    # two majorannas in pair
    assert np.isclose(psi[1, 1, 0, 0, 0, 0, 0, 0], 0)
    assert np.isclose(psi[0, 0, 0, 0, 0, 0, 1, 1], 0)
    # 3 qubits, 2 in pair, one not
    assert np.isclose(psi[0, 0, 1, 0, 0, 0, 1, 1], 0)

    assert np.isclose(psi[1, 0, 1, 0, 1, 0, 1, 0], -1)
    assert np.isclose(psi[0, 1, 0, 1, 0, 1, 0, 1], -1)

    assert np.isclose(psi[1, 1, 1, 1, 1, 1, 1, 1], 1)
    assert np.isclose(psi[1, 1, 1, 1, 1, 1, 0, 0], 0)


def get_majorana_operators(d):
    fs, fdags = _get_fs_fdags(d)

    ms = []

    for i in range(d):
        ms.append((fs[i] + fdags[i]))
        ms.append((fs[i] - fdags[i]) * (-1j))

    return ms


def test_psi_systematic():
    psi_tensor = get_psi()
    ms = get_majorana_operators(4)

    ket0 = np.array([1, 0])
    ket1 = np.array([0, 1])
    psi = (
        tensor_product([ket0, ket0, ket1, ket1])
        + tensor_product([ket1, ket1, ket0, ket0])
    ) / np.sqrt(2)

    for idx in range(2**8):
        tensor_index = np.empty(shape=2 * 4, dtype=int)
        for i in range(2 * 4):
            tensor_index[2 * 4 - i - 1] = idx % (2 * 1)
            idx = idx // (2 * 1)

        act_inds = np.where(tensor_index == 1)[0]
        filtered_ms = [ms[i] for i in act_inds]
        if len(filtered_ms) > 1:
            assert np.isclose(
                psi_tensor[tuple(tensor_index)],
                psi @ np.linalg.multi_dot(filtered_ms) @ psi,
            )


def get_prefactor(tensor_index):
    # first sort the factors and get the parity of the permutation
    sorted_tensor_index, prefactor = sort_and_get_parity(np.copy(tensor_index))
    # remove those completely, that appear an even number of times
    filtered_tensor_index, counts = np.unique(sorted_tensor_index, return_counts=True)
    filtered_tensor_index = np.delete(filtered_tensor_index, np.where(counts % 2 == 0))
    # if two majorannas are in pair, introduce a a factor of j and remove them
    del_indices = []
    for k in range(len(filtered_tensor_index) - 1):
        if (
            filtered_tensor_index[k] % 2 == 0
            and filtered_tensor_index[k + 1] == filtered_tensor_index[k] + 1
        ):
            del_indices += [k, k + 1]
            prefactor *= 1j
    filtered_tensor_index = np.delete(filtered_tensor_index, del_indices)
    # if we still have elements in the index list, then the factor should be 0
    if len(filtered_tensor_index) > 0:
        prefactor = 0

    return prefactor


def reduce_and_sort_majoranas(tensor_index):
    # first sort the factors and get the parity of the permutation
    sorted_tensor_index, prefactor = sort_and_get_parity(np.copy(tensor_index))
    filtered_tensor_index, counts = np.unique(sorted_tensor_index, return_counts=True)
    # remove those completely, that appear an even number of times
    filtered_tensor_index = np.delete(filtered_tensor_index, np.where(counts % 2 == 0))

    return filtered_tensor_index, prefactor


if __name__ == "__main__":
    N = 2
    l = 2

    d = 4 * N

    initial_vector= get_fermion_sampling_initial_vector(N=N, l=l)
    breakpoint()

    import piquasso as pq

    from scipy.stats import unitary_group

    simulator = pq.fermionic.HeisenbergSimulator(d=d)

    U = unitary_group.rvs(d)

    program = pq.Program(
        [pq.fermionic.Correlations(initial_tensor), pq.Interferometer(U)]
    )

    result = simulator.execute(program)

    
