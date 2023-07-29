import numpy as np
import scipy
import math

"""
x - position operator
p - momentum operator
    Based on Wikipedia:
    x = sqrt(hbar/(2*m*w))*(ad+a)
    p = sqrt(hbar*m*w/2)*(ad-a)
Both can be thought of as a matrix (based on ad, and a), and therefore:
We wish to generate self-adjoint "polynomials" H(x, p).
As an example for a rank 2: H(x, p) := a*xx + b*xp + c*px + d*pp + z
Since [x, p] = i, and the conjugate of xp = px
A good solution is when b equals the conjugate of c, and a, d, z are all real numbers.

Storing the polynomials:
One of the most important aspect of this is for it to be consistent as the degree of the polynomial increases.
We need to be able to tell for each polynomial if it's self-adjoint or not.
    This is possible if, for each coefficient, we can tell which is its conjugate counterpart.

The idea: Binary trees.
Since x, and p are not commutative, we need a systematic approach for constructing every possible ordering.
For degree of 4 some of the possibilities are: xxxx, xxxp, xppx, pppx, etc...
A binary tree is perfect for such a task, becasue if we go "left" in it, we put an x, otherwise a p.
This way, every combination has its own index, and using binary search we can query it's conjugate counterpart easily.
For a rank 3, we need binary trees of height from 0-3, and then adding them up to form the full polynomial.
    // 0 height trees "represent" the constant
Let's agree that the "left" in the binary tree corresponds to the value `True` and to the x operator.
Generating the polynomial:
    1.) Divide the problem into subproblems for rank n. Repeat while n > 1:
        - Generate every coefficient for only the n-th degree terms. // `generate_polynomial_of_rank`
        - For symmetric terms, the coefficients are real, for the others each has a pair which must be its conjugate.
            . Conjugate pairs are calculated with binary search, and then traversed the tree in the reversed order.
            . `find_conjugate_index` first calls `find_index_path` which uses binary search, and returns the path to be reversed.
        - n := n-1.
    2.) Add up the subproblems to form the polynomial coefficients. `generate_polynomial`
    3.) Construct the Hemiltonian based on the x, p operators, and the coefficients. // `calculate_polynomial_term_matrix`
        - Additional checks for self-adjointness can be done. // `is_self_adjoint_coeffs` or `gate - np.conj(gate).T`
    4.) Construct the gate. // `create_quantum_gate`

Note, in the code, the degrees are increasing


creation and annihilation operators, same thing.
adagger in front, a back, check self-adjoint
"""

rank = 6
cutoff = 3
hbar = 2.0
m = 1
omega = 1


def create_creation_operator_matrix(size: int) -> np.ndarray:
    elements = np.sqrt(np.arange(1, size))
    return np.diag(elements, -1)


def create_annihilation_operator_matrix(size: int) -> np.ndarray:
    elements = np.sqrt(np.arange(1, size))
    return np.diag(elements, 1)


def create_position_operator_matrix(size: int) -> np.ndarray:
    coeff = np.sqrt(hbar / (2 * m * omega))
    return coeff * (
        create_creation_operator_matrix(size)
        + create_annihilation_operator_matrix(size)
    )


def create_momentum_operator_matrix(size: int) -> np.ndarray:
    coeff = 1j * np.sqrt(hbar / (2 * m * omega))
    return coeff * (
        create_creation_operator_matrix(size)
        - create_annihilation_operator_matrix(size)
    )


def find_index_path(index: int, rank: int) -> list:
    start = 0
    end = 2**rank - 1
    direction_list = []

    while start <= end:

        if end - start == 1:
            direction_list.append(index == start)
            return direction_list

        midpoint = math.floor((start + end) / 2)

        if index > midpoint:
            start = midpoint + 1
            direction_list.append(False)
        else:
            end = midpoint - 1
            direction_list.append(True)

    return direction_list


def find_conjugate_index(index: int, rank: int) -> int:
    if index < 0 or index > 2**rank:
        print("Wrong index")
        return -1

    if rank < 2:
        return index

    direction_list = find_index_path(index, rank)
    start = 0
    end = 2**rank - 1
    direction_list.reverse()

    for i in range(rank):
        midpoint = math.floor((start + end) / 2)

        if direction_list[i]:
            end = midpoint
        else:
            start = midpoint

    if end - start == 1 and not direction_list[i]:
        # At the last step, the midpoint is always the start index, regardless of the last turn
        return midpoint + 1

    return midpoint


def generate_polynomial_of_rank(rank: int):
    # NOTE: Rank indexing does not start from 0. For future usage see `generate_polynomial`
    coeff_amount = 2 ** (rank)
    coeff_array = np.sqrt(np.random.uniform(0, 1, coeff_amount)) * np.exp(
        1.0j * np.random.uniform(0, 2 * np.pi, coeff_amount)
    )

    for j in range(coeff_amount):
        conjugate_index = find_conjugate_index(j, rank)

        if j == conjugate_index:
            coeff_array[j] = np.random.default_rng().random()
        if coeff_array[j] != np.conjugate(coeff_array[conjugate_index]):
            coeff_array[j] = np.conjugate(coeff_array[conjugate_index])

    return coeff_array


def generate_polynomial(rank: int) -> list:  # Extreme prototype
    full_coeffs = []
    full_coeffs.append(np.array(np.random.default_rng().random()))  # Constant term

    for i in range(1, rank - 1):
        full_coeffs.append(generate_polynomial_of_rank(i + 1))

    return full_coeffs


def is_self_adjoint_coeffs(coeffs: list) -> bool:
    bintree_amount = 1

    for idx, array in enumerate(coeffs):

        if idx == 0:
            if array != np.conjugate(array):
                return False  # len() failes on ndarray of size 1, and this is the constant value
            continue

        for i in range(len(array)):
            conjugate_index = find_conjugate_index(i, bintree_amount)
            if array[i] != np.conjugate(array[conjugate_index]):
                return False

        bintree_amount += 1

    return True


def calculate_polynomial_term_matrix(
    coefficient,
    index: int,
    rank: int,
    position_op: np.ndarray = None,
    momentum_op: np.ndarray = None,
) -> np.ndarray:
    if position_op is None:
        position_op = create_position_operator_matrix(cutoff)
    if momentum_op is None:
        momentum_op = create_momentum_operator_matrix(cutoff)

    direction_list = find_index_path(index, rank)
    term = np.identity(cutoff)

    for turn in direction_list:
        if turn:
            term = term @ position_op
        else:
            term = term @ momentum_op

    return coefficient * term


def create_quantum_gate(coeffs: list) -> np.ndarray:
    gate = np.zeros((cutoff, cutoff), dtype=np.complex128)

    if not is_self_adjoint_coeffs(coeffs):
        return gate

    for i, array in enumerate(coeffs):
        if i == 0:  # 0d array error
            gate += array
        else:
            for j, coeff in enumerate(array):
                gate += calculate_polynomial_term_matrix(coeff, j, i)

    assert np.all(np.isclose(gate - np.conj(gate).T, np.zeros(cutoff)))

    return scipy.linalg.expm(1j * gate)


# Dummy debug function for printing
def create_combinations(rank: int, perm: str):
    if rank == 0:
        symmetric = True
        for i in range(int(len(perm) / 2)):
            symmetric = symmetric and perm[i] == perm[len(perm) - i - 1]
        if symmetric:
            # print(create_combinations.index, perm)
            create_combinations.symm_count += 1
        create_combinations.index += 1
        print(create_combinations.index, perm)
    else:
        create_combinations(rank - 1, perm + "x")
        create_combinations(rank - 1, perm + "p")


create_combinations.index = 0
create_combinations.symm_count = 0
create_combinations(rank, "")
print(create_combinations.symm_count)
# print(is_self_adjoint_coeffs(generate_polynomial(rank)))
gate = create_quantum_gate(generate_polynomial(rank))
print(gate)
