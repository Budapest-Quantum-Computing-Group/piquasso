#
# Copyright 2021-2025 Budapest Quantum Computing Group
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


from functools import partial
from typing import Dict, Optional, List, Tuple, TYPE_CHECKING
import numpy as np

from piquasso._math.fock import cutoff_fock_space_dim, get_fock_space_basis
from piquasso._math.linalg import is_unitary

from piquasso.api.config import Config
from piquasso.api.state import State
from piquasso.api.exceptions import (
    InvalidState,
    PiquassoException,
    NotImplementedCalculation,
)
from piquasso.api.connector import BaseConnector

from .utils import calculate_state_vector, calculate_inner_product

if TYPE_CHECKING:
    import piquasso


class SamplingState(State):
    r"""A state dedicated for Boson Sampling (or related) simulations.

    When using :class:`~piquasso._simulators.sampling.simulator.SamplingSimulator`, the
    simulation results will contain an instance of this class, containing the input
    occupation numbers and the interferometer to be applied. Moreover, postselections
    and losses can also be specified in this state.

    Consider the input state

    .. math::
        \ket{\psi} = \sum_i c_i \ket{\mathbf{n}^{(i)}},

    where :math:`\ket{\mathbf{n}^{(i)}}` are Fock states defined on :math:`d` modes, and
    :math:`c_i` are the corresponding coefficients.
    Consider now a possibly lossy passive linear circuit :math:`\mathcal{C}` modeled by
    the quantum channel :math:`\Lambda_{\mathcal{C}}`, resulting in the output state

    .. math::
        \rho_{\text{out}} = \Lambda_{\mathcal{C}}(\ketbra{\psi}{\psi}).

    Optionally, certain modes can be postselected to specific photon counts
    :math:`\mathbf{m}`, resulting in the final (possibly unnormalized) state

    .. math::
        \rho_{\text{postselected}} = \mathrm{Tr}_{\text{postselected modes}} \left[
            (I \otimes \ketbra{\mathbf{m}}{\mathbf{m}}) \,
            \rho_{\text{out}}
        \right],

    where :math:`I \otimes \ketbra{\mathbf{m}}{\mathbf{m}}` is the projector
    corresponding to the postselection.

    Losses can be specified by :class:`~piquasso.instructions.channels.Loss`, and post-
    selections by :class:`~piquasso.instructions.measurements.PostSelectPhotons`.

    Example usage:

    .. code-block:: python

        >>> import piquasso as pq
        >>>
        >>> from scipy.stats import unitary_group
        >>>
        >>> d = 7
        >>>
        >>> interferometer_matrix = unitary_group.rvs(d)
        >>>
        >>> with pq.Program() as program:
        >>>     pq.Q(all) | pq.StateVector([1, 1, 1, 0, 0, 0, 0])
        >>>     pq.Q(all) | pq.Interferometer(interferometer_matrix)
        >>>     pq.Q(all) | pq.ParticleNumberMeasurement()
        >>>
        >>> simulator = pq.SamplingSimulator(d=d)
        >>>
        >>> result = simulator.execute(program, shots=3)
        >>>
        >>> result.samples
        [(0, 0, 0, 0, 2, 1, 0), (3, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 2, 1)]
    """

    def __init__(
        self, d: int, connector: BaseConnector, config: Optional[Config] = None
    ) -> None:
        """
        Args:
            d (int): The number of modes.
            connector (BaseConnector): Instance containing calculation functions.
            config (Config): Instance containing constants for the simulation.
        """
        super().__init__(connector=connector, config=config)

        self._occupation_numbers: List = []
        self._coefficients: List = []

        self.interferometer = np.diag(np.ones(d, dtype=self._config.complex_dtype))
        """The interferometer matrix corresponding to the circuit."""

        self.is_lossy = False
        """Returns `True` if the state is lossy, otherwise `False`."""

        self._postselections: Dict[int, int] = {}
        """Maps postselected modes to postselected photon counts."""

    def _set_postselection(
        self, modes: Tuple[int, ...], photon_counts: Tuple[int, ...]
    ) -> None:
        postselections = self._postselections

        # NOTE: This is tricky, since we need to be aware that the simulator filters
        # out the modes that are already measured - so we need to map the postselected
        # modes to the actual modes of the state.
        actual_modes = np.array(self._get_active_modes())[modes,]

        for i in range(len(actual_modes)):
            postselections[actual_modes[i]] = photon_counts[i]

        self._config.cutoff -= sum(photon_counts)
        self._postselections = postselections

    def _is_postselected(self) -> bool:
        return self._postselections != {}

    def _get_postselected_modes(self) -> Tuple[int, ...]:
        return tuple(self._postselections.keys())

    def _get_postselected_photons(self) -> Tuple[int, ...]:
        return tuple(self._postselections.values())

    def _get_active_modes(self) -> Tuple[int, ...]:
        postselected_modes = self._get_postselected_modes()
        return tuple(
            [
                i
                for i in range(self.total_number_of_modes)
                if i not in postselected_modes
            ]
        )

    @property
    def total_number_of_modes(self) -> int:
        """The total number of modes, including the postselected ones."""
        return len(self.interferometer)

    def validate(self) -> None:
        """Validates the current state.

        Raises:
            InvalidState: If the interferometer matrix is non-unitary, or the input
                state is invalid.
        """

        if not is_unitary(self.interferometer):
            raise InvalidState("The interferometer matrix is not unitary.")

        for occupation_number in self._occupation_numbers:
            if len(occupation_number) != self.total_number_of_modes:
                raise InvalidState(
                    f"The occupation number '{occupation_number}' is not "
                    f"well-defined on '{self.total_number_of_modes}' modes."
                )

        norm = self.norm

        if not np.isclose(norm, 1.0):
            raise InvalidState(f"The state is not normalized: norm={norm}")

    @property
    def d(self):
        return len(self.interferometer) - len(self._get_postselected_modes())

    @property
    def norm(self):
        """The norm of the state."""
        np = self._connector.np
        if self._is_postselected():
            return np.sum(self.fock_probabilities)

        return np.sum(np.abs(self._coefficients) ** 2)

    def normalize(self) -> None:
        """Normalizes the state.

        Raises:
            InvalidState: If the norm of the state is 0.
        """
        norm = self.norm

        if self._config.validate and np.isclose(norm, 0):
            raise InvalidState("The norm of the state is 0.")

        self._coefficients = [c / np.sqrt(norm) for c in self._coefficients]

    def get_particle_detection_probability(
        self, occupation_number: np.ndarray
    ) -> float:
        if self._config.validate and len(occupation_number) != self.d:
            raise PiquassoException(
                f"The specified occupation number should have length '{self.d}': "
                f"occupation_number='{occupation_number}'. "
                f"Remark: The modes {self._get_postselected_modes()} have been "
                f"postselected."
            )

        np = self._connector.np
        fallback_np = self._connector.fallback_np

        output_number_of_particles = fallback_np.sum(occupation_number) + sum(
            self._postselections.values()
        )

        sum_ = 0.0

        postselected_modes = self._get_postselected_modes()
        active_modes = self._get_active_modes()
        postselected_photons = self._get_postselected_photons()

        for index, input_occupation_number in enumerate(self._occupation_numbers):
            if output_number_of_particles != fallback_np.sum(input_occupation_number):
                continue

            full_occupation_number = fallback_np.zeros(
                self.total_number_of_modes, dtype=int
            )

            full_occupation_number[active_modes,] = occupation_number
            full_occupation_number[postselected_modes,] = postselected_photons

            inner_product = calculate_inner_product(
                interferometer=self.interferometer,
                input=input_occupation_number,
                output=full_occupation_number,
                connector=self._connector,
            )
            coefficient = self._coefficients[index]

            sum_ += coefficient * inner_product

        return np.abs(sum_) ** 2

    @property
    def state_vector(self):
        """The state vector representation of this state.

        The state vector is calculated in the truncated Fock space, taking into account
        any postselections that may have been applied. Moreover, the ordering of the
        Fock basis is increasing with particle numbers, and in each particle number
        conserving subspace, anti-lexicographic ordering is used.

        Raises:
            NotImplementedCalculation: If the state is lossy.
        """  # noqa: E501

        if self.is_lossy:
            raise NotImplementedCalculation(
                "This property is not implemented for lossy states. If you need it, "
                "please create an issue at "
                "https://github.com/Budapest-Quantum-Computing-Group/piquasso/issues."
            )

        connector = self._connector
        np = connector.np
        fallback_np = connector.fallback_np

        postselected_modes = self._get_postselected_modes()
        postselected_photons = self._get_postselected_photons()

        postselected_no_of_photons = fallback_np.sum(
            self._get_postselected_photons(), dtype=int
        )

        calculate_state_vector_func = partial(
            calculate_state_vector,
            interferometer=self.interferometer,
            postselect_data=(postselected_modes, postselected_photons),
            config=self._config,
            connector=self._connector,
        )

        particle_numbers = fallback_np.sum(self._occupation_numbers, axis=1)

        relevant_indices = fallback_np.where(
            particle_numbers < postselected_no_of_photons
        )[0]

        particle_numbers = fallback_np.delete(
            particle_numbers, relevant_indices, axis=0
        )
        occupation_numbers = fallback_np.delete(
            self._occupation_numbers, relevant_indices, axis=0
        )

        state_vector = np.zeros(
            shape=cutoff_fock_space_dim(d=self.d, cutoff=self._config.cutoff),
            dtype=self._config.complex_dtype,
        )

        for index in range(len(occupation_numbers)):
            particle_number = particle_numbers[index]

            starting_index = cutoff_fock_space_dim(
                d=self.d,
                cutoff=particle_number - postselected_no_of_photons,
            )

            ending_index = cutoff_fock_space_dim(
                d=self.d,
                cutoff=particle_number - postselected_no_of_photons + 1,
            )
            coefficient = self._coefficients[index]

            input_state = self._occupation_numbers[index]

            partial_state_vector = calculate_state_vector_func(
                initial_state=input_state,
            )

            index_range = fallback_np.arange(starting_index, ending_index)

            state_vector = connector.assign(
                state_vector,
                index_range,
                state_vector[index_range] + coefficient * partial_state_vector,
            )

        return state_vector

    @property
    def state_vector_map(self) -> Dict[Tuple[int, ...], complex]:
        map_: Dict[Tuple[int, ...], complex] = {}

        space = get_fock_space_basis(d=self.d, cutoff=self._config.cutoff)

        state_vector = self.state_vector

        for index, basis in enumerate(space):
            map_[tuple(basis)] = state_vector[index]

        return map_

    @property
    def density_matrix(self) -> np.ndarray:
        """The density matrix of the state in the truncated Fock space."""
        state_vector = self.state_vector

        return self._np.outer(state_vector, self._np.conj(state_vector))

    @property
    def fock_probabilities(self) -> np.ndarray:
        """The Fock basis probabilities of the state."""
        np = self._connector.np

        state_vector = self.state_vector

        return np.real(np.conj(state_vector) * state_vector)

    @property
    def fock_probabilities_map(self) -> Dict[Tuple[int, ...], float]:
        probability_map: Dict[Tuple[int, ...], float] = {}

        space = get_fock_space_basis(d=self.d, cutoff=self._config.cutoff)

        state_vector = self.state_vector

        for index, basis in enumerate(space):
            probability_map[tuple(basis)] = self._np.abs(state_vector[index]) ** 2

        return probability_map

    def to_pure_fock_state(
        self,
    ) -> "piquasso._simulators.fock.pure.state.PureFockState":
        """Converts this state to a pure Fock state.

        Returns:
            piquasso._simulators.fock.pure.state.PureFockState:
                The corresponding pure Fock state.
        """
        from piquasso._simulators.fock.pure.state import PureFockState

        pure_fock_state = PureFockState(
            d=self.d,
            connector=self._connector,
            config=self._config,
        )

        pure_fock_state.state_vector = self.state_vector

        return pure_fock_state

    def get_purity(self) -> float:
        """Returns the purity of the state.

        Returns:
            float: The purity of the state.
        """

        if not self.is_lossy:
            return 1.0

        raise NotImplementedCalculation(
            "Purity calculation is not implemented for lossy states."
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SamplingState):
            return False

        return (
            np.allclose(self._coefficients, other._coefficients)
            and np.allclose(self._occupation_numbers, other._occupation_numbers)
            and np.allclose(self.interferometer, other.interferometer)
            and self.is_lossy == other.is_lossy
            and self._postselections == other._postselections
        )
