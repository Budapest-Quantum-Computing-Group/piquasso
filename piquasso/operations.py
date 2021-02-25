#
# Copyright (C) 2020 by TODO - All rights reserved.
#

"""Simple passive linear optical elements."""

import numpy as np

from piquasso import registry
from piquasso.context import Context

from piquasso.gaussian.backend import GaussianBackend
from piquasso.sampling.backend import SamplingBackend


class Operation(registry.ClassRecorder):
    backends = {}

    def __init__(self, *params):
        self.params = params
        self.modes = None

    @classmethod
    def from_properties(cls, properties):
        """Creates an `Operation` instance from a mapping specified.

        Overrides `registry.ClassRecorder.from_properties`, since the `modes` attribute
        of an `Operator` instance cannot be specified in `__init__`, it has to be
        specified after initialization.

        Args:
            properties (collections.Mapping):
                The desired `Operator` instance in the format of a mapping.

        Returns:
            Operator: An `Operator` initialized using the specified mapping.
        """

        operation = cls(**properties["params"])

        operation.modes = properties["modes"]

        return operation

    def __call__(self, backend):
        """Executes the operation on the specified backend.

        Args:
            backend (Backend): The backend to execute the operation on.
        """

        method = self._resolve_method(backend)

        method(backend, self)

    def _resolve_method(self, backend):
        """Resolves the method according to the specified `backend` instance.

        Args:
            backend (Backend): The `Backend` on which the desired method is defined.

        Raises:
            NotImplementedError: If no such method is implemented on the `Backend`.

        Returns:
            The method which corresponds to the operation on `Backend`.
        """
        method = self.backends.get(backend.__class__)

        if not method:
            raise NotImplementedError("No such operation implemented on this backend.")

        return method


class ModelessOperation(Operation):
    def __init__(self, *params):
        super().__init__(*params)

        Context.current_program.operations.append(self)


class PassiveTransform(Operation):
    r"""Applies a general passive transformation.

    Args:
        T (np.array):
            The representation of the passive transformation on the one-particle
            subspace.
    """

    backends = {
        GaussianBackend: GaussianBackend._apply_passive,
    }

    def __init__(self, T):
        self._passive_representation = T


class B(Operation):
    r"""Applies a beamsplitter operation.

    The matrix representation of the beamsplitter operation
    is

    .. math::
        B = \begin{bmatrix}
            t  & r^* \\
            -r & t
        \end{bmatrix},

    where :math:`t = \cos(\theta)` and :math:`r = e^{- i \phi} \sin(\theta)`.

    """

    backends = {
        GaussianBackend: GaussianBackend._apply_passive,
        SamplingBackend: SamplingBackend._multiply_interferometer,
    }

    def __init__(self, theta=0., phi=np.pi / 4):
        r"""Beamsplitter operation

        Args:
            phi (float): Phase angle of the beamsplitter.
                (defaults to :math:`\phi = \pi/2` that gives a symmetric beamsplitter)
            theta (float): The transmittivity angle of the beamsplitter.
                (defaults to :math:`\theta=\pi/4` that gives a 50-50 beamsplitter)
        """
        super().__init__(theta, phi)

        self._passive_representation = self._get_passive_representation()

    def _get_passive_representation(self):
        theta, phi = self.params

        t = np.cos(theta)
        r = np.exp(-1j * phi) * np.sin(theta)

        return np.array([
            [t, np.conj(r)],
            [-r, t]
        ])


class R(Operation):
    r"""Rotation or Phaseshifter operation.

    The annihilation and creation operators are evolved in the following
    way:

    .. math::
        P(\phi) \hat{a}_k P(\phi)^\dagger = e^{i \phi} \hat{a}_k \\
        P(\phi) \hat{a}_k^\dagger P(\phi)^\dagger
            = e^{- i \phi} \hat{a}_k^\dagger

    """

    backends = {
        GaussianBackend: GaussianBackend._apply_passive,
        SamplingBackend: SamplingBackend._multiply_interferometer,
    }

    def __init__(self, phi):
        r"""Rotation or Phaseshifter operation.

        Args:
            phi (float): The angle of the rotation.
        """
        super().__init__(phi)

        self._passive_representation = self._get_passive_representation()

    def _get_passive_representation(self):
        phi = self.params[0]
        phase = np.exp(1j * phi)

        return np.array([[phase]])


class S(Operation):
    r"""Applies the squeezing operator.

    The definition of the operator is:

    .. math::
            S(z) = \exp\left(\frac{1}{2}(z^* a^2 -z {a^\dagger}^2)\right)

    where :math:`z = r e^{i\theta}`. The :math:`r` parameter is the amplitude of the squeezing
    and :math:`\theta` is the angle of the squeezing.

    This act of squeezing at a given rotation angle :math:`\theta` results in a shrinkage in the
    :math:`\hat{x}` quadrature and a stretching in the other quadrature :math:`\hat{p}` as follows:

    .. math::
        S^\dagger(z) x_{\theta} S(z) = e^{-r} x_{\theta}, \: S^\dagger(z) p_{\theta} S(z) = e^{r} p_{\theta}

    The action of the :math:`\hat{S}(z)` gate on the ladder operators :math:`\hat{a}`
    and :math:`\hat{a}^\dagger` can be defined as follows:

    .. math::
        {S(z)}^{\dagger}\hat{a}S(z) = \alpha\hat{a} - \beta \hat{a}^{\dagger} \\
            {S(z)}^{\dagger}\hat{a}^\dagger S(z) = \alpha\hat{a}^\dagger - \beta^* \hat{a}

    where :math:`\alpha` and :math:`\beta` are :math:`\cosh(amp)`, :math:`e^{i\theta}\sinh(amp)`
    respectively.
    """  # noqa: E501

    backends = {
        GaussianBackend: GaussianBackend.squeezing,
    }

    def __init__(self, amp, theta=0):
        r"""
        Squeezing operation.

        The Hamiltonian of this operator is defined in terms of `z`:

        .. math:
            z = amp \exp(i\theta)

        Args:
            amp (float): The amplitude of the squeezing operation.
            theta (float): The squeezing angle.
        """

        super().__init__(amp, theta)


class P(Operation):
    r"""Applies the quadratic phase operation to the state.

    The operator of the quadratic phase gate is

    .. math::
        P(s) = \exp (i \frac{s \hat{x}}{2\hbar}),

    and it evolves the annihilation operator as

    .. math::
        P(s)^\dagger a_i P(s) = (1 + i \frac{s}{2}) a_i + i \frac{s}{2} a_i^\dagger.
    """

    backends = {
        GaussianBackend: GaussianBackend.quadratic_phase,
    }

    def __init__(self, s):
        super().__init__(s)


class D(Operation):
    """Displacement operation."""

    backends = {
        GaussianBackend: GaussianBackend.displacement,
    }

    def __init__(self, *, alpha=None, r=None, phi=None):
        r"""Phase space displacement operation.

        One must either specify :param:`alpha` only, or the combination of :param:`r`
        and :param:`phi`.

        When :param:`r` and :param:`phi` are the given parameters, `alpha` is calculated
        via:

        .. math:
            \alpha = r \exp(i \phi)

        Args:
            alpha (complex): The displacement.
            r (float): The displacement magnitude.
            phi (float): The displacement angle.
        """
        assert \
            alpha is not None and r is None and phi is None \
            or \
            alpha is None and r is not None and phi is not None, \
            "Either specify 'alpha' only, or the combination of 'r' and 'phi'."

        if alpha is None:
            alpha = r * np.exp(1j * phi)

        super().__init__(alpha)


class Interferometer(Operation):
    """Interferometer.

    Adds additional interferometer to the effective interferometer.

    This can be interpreted as placing another interferometer in the network, just
    before performing the sampling. This operation is realized by multiplying
    current effective interferometer matrix with new interferometer matrix.

    Do note, that new interferometer matrix works as interferometer matrix on
    qumodes (provided as the arguments) and as an identity on every other mode.
    """

    backends = {
        SamplingBackend: SamplingBackend.interferometer
    }

    @staticmethod
    def _is_square(matrix):
        shape = matrix.shape
        return len(shape) == 2 and shape[0] == shape[1]

    def __init__(self, interferometer_matrix):
        r"""Interferometer

        Args:
            interferometer_matrix (np.array): A square matrix representing the
                interferometer
        """
        assert \
            self._is_square(interferometer_matrix), \
            "The interferometer matrix should be a square matrix."
        super().__init__(interferometer_matrix)


class Sampling(ModelessOperation):
    r"""Boson Sampling.

    Simulates a boson sampling using generalized Clifford&Clifford algorithm
    from [Brod, Oszmaniec 2020].

    This method assumes that initial_state is given in the second quantization
    description (mode occupation). BoSS requires input states as numpy arrays,
    therefore the state is prepared as such structure.

    Generalized Cliffords simulation strategy form [Brod, Oszmaniec 2020] was used
    as it allows effective simulation of broader range of input states than original
    algorithm.

    Args:
        shots (int): A positive integer value representing number of samples for the
            experiment
    """

    backends = {
        SamplingBackend: SamplingBackend.sampling
    }

    def __init__(self, shots=1):
        assert \
            shots > 0 and isinstance(shots, int),\
            "The number of shots should be a positive integer."
        super().__init__(shots)
