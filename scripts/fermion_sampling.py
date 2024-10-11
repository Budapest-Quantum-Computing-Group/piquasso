import numpy as np
import scipy
from piquasso._math.combinatorics import sort_and_get_parity
from piquasso.fermionic.gaussian._misc import _get_fs_fdags, tensor_product
import piquasso as pq

connector = pq.NumpyConnector()


def get_fermion_sampling_initial_tensor(N: int, l: int):
    tensor = np.empty(shape=(2 * N,) * (2 * l), dtype=complex)
    size = (2 * N) ** (2 * l)
    psi_tensor = get_psi()
    for idx in range(size):
        tensor_index = np.empty(shape=2 * l)
        for i in range(2 * l):
            tensor_index[2 * l - i - 1] = idx % (2 * N)
            idx = idx // (2 * N)
        first_quantized_index = np.arange(2 * N)[tensor_index == 1]


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


def reduce_and_sort_majoranas(tensor_index, l):
    sorted_tensor_index, prefactor = sort_and_get_parity(np.copy(tensor_index))
    filtered_tensor_index, counts = np.unique(sorted_tensor_index, return_counts=True)
    filtered_tensor_index = np.delete(filtered_tensor_index, np.where(counts % 2 == 0))
