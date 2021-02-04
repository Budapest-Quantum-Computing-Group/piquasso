#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import itertools
import numpy as np

from scipy.special import factorial, comb


def direct_sum(*args):
    # TODO: Omit recursions!
    if len(args) == 2:
        a = args[0]
        b = args[1]
        dsum = np.zeros(np.add(a.shape, b.shape), dtype=complex)
        dsum[: a.shape[0], : a.shape[1]] = a
        dsum[a.shape[0]:, a.shape[1]:] = b

        return dsum

    return direct_sum(args[0], direct_sum(*args[1:]))


class FockBasis(np.ndarray):
    def __new__(cls, *args, **kwargs):
        kwargs["dtype"] = int
        return np.array(*args, **kwargs).view(cls)

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        return self.display(template="|{}>")

    def display(self, template):
        return template.format("".join([str(element) for element in self]))

    def display_as_bra(self):
        return self.display("<{}|")

    def __repr__(self):
        return str(self)

    @property
    def d(self):
        return len(self)

    @classmethod
    def create_all(cls, n, d):
        if n == 0:
            yield np.zeros(d, dtype=int).view(cls)

        else:
            masks = np.rot90(np.identity(d, dtype=int)).view(cls)

            for c in itertools.combinations_with_replacement(masks, n):
                yield sum(c)

    @property
    def first_quantized(self):
        ret = []

        for idx, repetition in enumerate(self):
            ret.extend([idx + 1] * repetition)

        return ret

    @property
    def all_possible_first_quantized_vectors(self):
        return list(set(itertools.permutations(self.first_quantized)))


class FockSpace:
    r"""
    Note, that when you symmetrize a tensor, i.e. use the superoperator

    .. math::
        S: A \otimes B \mapsto A \vee B

    on a tensor which is expressed in the regular Fock basis, then the resulting tensor
    still remains in the regular representation. You have to perform a basis
    transformation to acquire the symmetrized tensor in the symmetrized representation.
    """

    def __init__(self, *, d, cutoff):
        self.d = d
        self.cutoff = cutoff

        self._basis = self._get_all_basis()

    def get_fock_operator(self, operator):
        return direct_sum(
            *(
                self._symmetric_tensorpower(operator, n)
                for n in range(self.cutoff + 1)
            )
        )

    @property
    def cardinality(self):
        r"""
        ..math::
            \sum_{n=0}^i {d + n - 1 \choose n} = \frac{(i + 1) {d + i \choose i + 1}}{d}
        """
        return int(
            (self.cutoff + 1)
            * comb(self.d + self.cutoff, self.cutoff + 1)
            / self.d
        )

    def _get_all_basis(self):
        ret = []

        for n in range(self.cutoff + 1):
            ret.append(list(FockBasis.create_all(n, self.d)))

        return ret

    def _symmetric_cardinality(self, n):
        return int(comb(self.d + n - 1, n))

    def _symmetric_tensorpower(self, operator, n):
        if n == 0:
            return np.array([[1]])

        if n == 1:
            return operator

        ret = np.zeros(shape=(self._symmetric_cardinality(n), )*2, dtype=complex)

        basis = self._basis[n]

        for index1, basis_vector1 in enumerate(basis):
            for index2, basis_vector2 in enumerate(basis):

                sum_ = 0

                for permutation1 in basis_vector1.all_possible_first_quantized_vectors:
                    for permutation2 in (
                        basis_vector2.all_possible_first_quantized_vectors
                    ):
                        prod = 1

                        for i in range(len(permutation1)):
                            i1 = permutation1[i]
                            i2 = permutation2[i]
                            prod *= operator[i1 - 1, i2 - 1]

                        sum_ += prod

                normalization = (
                    np.power(
                        np.prod(list(factorial(basis_vector1)))
                        * np.prod(list(factorial(basis_vector2))),
                        1/2
                    ) / factorial(n)
                )

                ret[index1, index2] = normalization * sum_

        return ret

    def __repr__(self):
        return str(self._basis)
