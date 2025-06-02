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

from typing import Optional, Tuple, Dict

import numpy as np
from numpy.polynomial.hermite import hermval

from piquasso.api.config import Config
from piquasso.api.exceptions import InvalidState, PiquassoException
from piquasso.api.connector import BaseConnector

from piquasso._math.fock import cutoff_fock_space_dim, get_fock_space_basis
from piquasso._math.linalg import vector_absolute_square
from piquasso._math.indices import (
    get_index_in_fock_space,
    get_index_in_fock_space_array,
    get_auxiliary_modes,
)

from ..state import BaseFockState
from ..general.state import FockState


class PureFockState(BaseFockState):
    r"""A simulated pure Fock state.

    If no mixed states are needed for a Fock simulation, then this state is the most
    appropriate currently, since it does not store an entire density matrix, only a
    state vector with size

    .. math::
        {d + c - 1 \choose c - 1},

    where :math:`c \in \mathbb{N}` is the Fock space cutoff.

    :ivar state_vector: The state vector of the quantum state.
    """

    def __init__(
        self, *, d: int, connector: BaseConnector, config: Optional[Config] = None
    ) -> None:
        """
        Args:
            d (int): The number of modes.
            connector (BaseConnector): Instance containing calculation functions.
            config (Config): Instance containing constants for the simulation.
        """

        super().__init__(d=d, connector=connector, config=config)

        self.state_vector = self._get_empty()

    def _get_empty_list(self) -> list:
        state_vector_size = cutoff_fock_space_dim(cutoff=self._config.cutoff, d=self.d)
        return [0.0] * state_vector_size

    def _get_empty(self) -> np.ndarray:
        state_vector_size = cutoff_fock_space_dim(cutoff=self._config.cutoff, d=self.d)

        return self._np.zeros(
            shape=(state_vector_size,), dtype=self._config.complex_dtype
        )

    def reset(self) -> None:
        state_vector_list = self._get_empty_list()
        state_vector_list[0] = 1.0

        self.state_vector = self._np.array(
            state_vector_list, dtype=self._config.complex_dtype
        )

    def _nonzero_elements_for_single_state_vector(self, state_vector):
        np = self._connector.np
        nonzero_indices = np.nonzero(state_vector)[0]

        occupation_numbers = self._space[nonzero_indices]

        nonzero_elements = state_vector[nonzero_indices]

        for index, coefficient in enumerate(nonzero_elements):
            yield coefficient, tuple(occupation_numbers[index])

    @property
    def nonzero_elements(self):
        return self._nonzero_elements_for_single_state_vector(self.state_vector)

    def _get_repr_for_single_state_vector(self, nonzero_elements):
        return " + ".join(
            [str(coefficient) + str(basis) for coefficient, basis in nonzero_elements]
        )

    def __str__(self) -> str:
        return self._get_repr_for_single_state_vector(self.nonzero_elements)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PureFockState):
            return False
        return self._np.allclose(self.state_vector, other.state_vector)

    @property
    def density_matrix(self) -> np.ndarray:
        cardinality = cutoff_fock_space_dim(d=self.d, cutoff=self._config.cutoff)

        state_vector = self.state_vector[:cardinality]

        return self._np.outer(state_vector, self._np.conj(state_vector))

    def _as_mixed(self) -> FockState:
        return FockState.from_fock_state(self)

    def reduced(self, modes: Tuple[int, ...]) -> FockState:
        return self._as_mixed().reduced(modes)

    def get_particle_detection_probability(
        self, occupation_number: np.ndarray
    ) -> float:
        if self._config.validate and len(occupation_number) != self.d:
            raise PiquassoException(
                f"The specified occupation number should have length '{self.d}': "
                f"occupation_number='{occupation_number}'."
            )

        index = get_index_in_fock_space(occupation_number)

        return self._np.real(
            self.state_vector[index].conjugate() * self.state_vector[index]
        )

    def get_particle_detection_probability_on_modes(
        self,
        occupation_numbers: np.ndarray,
        modes: Tuple[int, ...],
    ) -> float:
        np = self._connector.np
        fallback_np = self._connector.fallback_np

        occupation_numbers = fallback_np.array(occupation_numbers)

        auxiliary_d = self.d - len(modes)
        auxiliary_cutoff = self._config.cutoff - sum(occupation_numbers)
        auxiliary_modes = get_auxiliary_modes(self.d, modes)

        unordered_modes = fallback_np.concatenate([modes, auxiliary_modes])

        card = cutoff_fock_space_dim(d=auxiliary_d, cutoff=auxiliary_cutoff)
        auxiliary_basis = get_fock_space_basis(d=auxiliary_d, cutoff=auxiliary_cutoff)

        repeated_occupation_numbers = fallback_np.repeat(
            occupation_numbers[None, :], card, axis=0
        )

        unordered_occupation_numbers = fallback_np.concatenate(
            [repeated_occupation_numbers, auxiliary_basis], axis=1
        )

        sorter = fallback_np.argsort(unordered_modes)

        ordered_occupation_numbers = unordered_occupation_numbers[:, sorter]

        indices = get_index_in_fock_space_array(ordered_occupation_numbers)

        return np.sum(np.abs(self.state_vector[indices]) ** 2)

    @property
    def fock_probabilities(self) -> np.ndarray:
        return vector_absolute_square(self.state_vector, self._connector)

    @property
    def fock_probabilities_map(self) -> Dict[Tuple[int, ...], float]:
        probability_map: Dict[Tuple[int, ...], float] = {}

        for index, basis in enumerate(self._space):
            probability_map[tuple(basis)] = self._np.abs(self.state_vector[index]) ** 2

        return probability_map

    def normalize(self) -> None:
        norm = self.norm

        if self._config.validate and np.isclose(norm, 0):
            raise InvalidState("The norm of the state is 0.")

        self.state_vector = self.state_vector / self._np.sqrt(norm)

    def validate(self) -> None:
        """Validates the represented state.

        Raises:
            InvalidState:
                Raised, if the norm of the state vector is not close to 1.0.
        """
        if not self._config.validate:
            return

        sum_of_probabilities = sum(self.fock_probabilities)

        if not self._np.isclose(sum_of_probabilities, 1.0):
            raise InvalidState(
                f"The sum of probabilities is {sum_of_probabilities}, "
                "instead of 1.0:\n"
                f"fock_probabilities={self.fock_probabilities}"
            )

    def _get_mean_position_indices(self, mode):
        fallback_np = self._connector.fallback_np

        self._space[:, mode] -= 1
        lowered_indices = get_index_in_fock_space_array(self._space)
        self._space[:, mode] += 2
        raised_indices = get_index_in_fock_space_array(self._space)
        self._space[:, mode] -= 1

        relevant_column = self._space[:, mode]

        nonzero_indices_on_mode = (relevant_column > 0).nonzero()[0]
        upper_index = cutoff_fock_space_dim(d=self.d, cutoff=self._config.cutoff - 1)

        multipliers = fallback_np.sqrt(
            fallback_np.concatenate(
                [
                    relevant_column[nonzero_indices_on_mode],
                    relevant_column[:upper_index] + 1,
                ]
            )
        )
        left_indices = fallback_np.concatenate(
            [lowered_indices[nonzero_indices_on_mode], raised_indices[:upper_index]]
        )
        right_indices = fallback_np.concatenate(
            [nonzero_indices_on_mode, fallback_np.arange(upper_index)]
        )

        return multipliers, left_indices, right_indices

    def mean_position(self, mode: int) -> np.ndarray:
        np = self._connector.np
        fallback_np = self._connector.fallback_np
        multipliers, left_indices, right_indices = self._get_mean_position_indices(mode)

        state_vector = self.state_vector

        accumulator = np.dot(
            (multipliers * state_vector[left_indices]),
            state_vector[right_indices],
        )

        return np.real(accumulator) * fallback_np.sqrt(self._config.hbar / 2)

    def quadratures_mean_variance(
        self, modes: Tuple[int, ...], phi: float = 0
    ) -> Tuple[float, float]:
        r"""This method calculates the mean and the variance of the qudrature operators
        for a single qumode state.
        The quadrature operators :math:`x` and :math:`p` for a mode :math:`i`
        can be calculated using the creation and annihilation operators as follows:

        .. math::
            x_i &= \sqrt{\frac{\hbar}{2}} (a_i + a_i^\dagger) \\
                p_i &= -i \sqrt{\frac{\hbar}{2}} (a_i - a_i^\dagger).

        Let :math:`\phi \in [ 0, 2 \pi )`, we can rotate the quadratures using the
        following transformation:

        .. math::
            Q_{i, \phi} = \cos\phi~x_i + \sin\phi~p_i.

        The expectation value :math:`\langle Q_{i, \phi}\rangle` can be calculated as:

        .. math::
            \operatorname{Tr}(\rho_i Q_{i, \phi}),

        where :math:`\rho_i` is the reduced density matrix of the mode :math:`i` and
        :math:`Q_{i, \phi}` is the rotated quadrature operator for a single mode and
        the variance is calculated as:

        .. math::
            \operatorname{\textit{Var}(Q_{i,\phi})} = \langle Q_{i, \phi}^{2}\rangle
                - \langle Q_{i, \phi}\rangle^{2}.

        Args:
            phi (float): The rotation angle. By default it is `0` which means that
                the mean of the position operator is being calculated. For :math:`\phi=
                \frac{\pi}{2}` the mean of the momentum operator is being calculated.
            modes (tuple[int]): The correspoding mode at which the mean of the
                quadratures are being calculated.
        Returns:
            (float, float): A tuple that contains the expectation value and the
                varianceof of the quadrature operator respectively.
        """
        np = self._connector.np

        reduced_dm = self.reduced(modes=modes).density_matrix
        annih = np.diag(np.sqrt(np.arange(1, self._config.cutoff)), 1)
        create = annih.T
        position = (create + annih) * np.sqrt(self._config.hbar / 2)
        momentum = -1j * (annih - create) * np.sqrt(self._config.hbar / 2)

        if phi != 0:
            rotated_quadratures = position * np.cos(phi) + momentum * np.sin(phi)
        else:
            rotated_quadratures = position

        expectation = np.real(np.trace(np.dot(reduced_dm, rotated_quadratures)))
        variance = (
            np.real(
                np.trace(
                    np.dot(reduced_dm, np.dot(rotated_quadratures, rotated_quadratures))
                )
            )
            - expectation**2
        )
        return expectation, variance

    def mean_photon_number(self):
        r"""Returns the mean photon number

        .. math::
            \mathbb{E}_{\ket{\psi}}(\hat{n}) :=
                \bra{\psi} \hat{n} \ket{\psi},

        where :math:`\hat{n}` is the total photon number operator.
        """

        numbers = np.sum(self._space, axis=1)

        return numbers @ (self._np.abs(self.state_vector) ** 2)

    def variance_photon_number(self):
        r"""Returns the photon number variance

        .. math::
            \operatorname{Var}_{\ket{\psi}}(\hat{n}) :=
                \bra{\psi} (\hat{n} - \bar{n})^2 \ket{\psi},

        where :math:`\hat{n}` is the total photon number operator and :math:`\bar{n}`
        is its expectation value given by :meth:`mean_photon_number`.
        """

        numbers = np.sum(self._space, axis=1)

        probabilities = self._np.abs(self.state_vector) ** 2

        mean = numbers @ probabilities

        return (numbers - mean) ** 2 @ probabilities

    def get_tensor_representation(self):
        cutoff = self._config.cutoff
        d = self.d

        return self._connector.scatter(
            self._space,
            self.state_vector,
            [cutoff] * d,
        )

    def copy(self) -> "PureFockState":
        # NOTE: `__deepcopy__` is not allowed for tensorflow variables, so we have to
        # do it explicitely here.
        state = self.__class__(
            d=self.d, connector=self._connector, config=self._config.copy()
        )

        state.state_vector = self._connector.np.copy(self.state_vector)

        return state

    def get_purity(self):
        return 1.0

    def plot_wigner_purefock(
        self,
        mode: int = 0,
        xlims: Optional[Tuple[float, float]] = None,
        plims: Optional[Tuple[float, float]] = None,
        resolution: int = 200,
        y_integration_range: float = 5.0,
        y_points: int = 200,
        library: Literal["matplotlib", "plotly"] = "matplotlib",
    ):
        r"""
        Plot the single‐mode Wigner function of a PureFockState by numerical integration.

        This uses
            W(x,p) = (1 / (π ℏ)) ∫_{-∞}^∞ e^{(2 i p y)/ℏ} ψ(x - y) ψ^*(x + y) dy,
        where
            ψ(q) = ∑_{n=0}^{c−1} c_n φ_n(q),
            φ_n(q) = (π ℏ)^(-1/4) (1 / sqrt{2^n n!}) H_n(q / sqrt{ℏ}) e^{−q^2 / (2ℏ)}.

        Parameters
        ----------
        state : PureFockState
            A Piquasso PureFockState (with some cutoff `c = state._config.cutoff`).
        mode : int, default=0
            Which mode index (0‐based) to visualize. Must satisfy 0 ≤ mode < state.d.
        xlims : (float, float) or None
            If provided, these are (x_min, x_max). Otherwise, defaults to ±4σ_x, where
            σ_x = √⟨x²⟩ for that single mode (computed from its reduced density).
        plims : (float, float) or None
            If provided, these are (p_min, p_max). Otherwise, defaults to ±4σ_p.
        resolution : int, default=200
            Number of grid points along *each* of the x and p axes
            (so the total grid is resolution×resolution).
        y_integration_range : float, default=5.0
            We truncate the †y†‐integral to [−y_range, +y_range].  If ℏ is small,
            you may need a larger range; if it's big or the state is narrow, smaller
            may suffice.
        y_points : int, default=200
            Number of points to discretize the †y†‐integral on [−y_range, +y_range].
        library : {"matplotlib", "plotly"}
            Which plotting backend to attempt. If it's not installed, an ImportError
            will be raised with a user‐friendly message.

        Returns
        -------
        - If `library="matplotlib"`, returns `(fig, ax)`.
        - If `library="plotly"`, returns a `plotly.graph_objects.Figure`.

        Raises
        ------
        ImportError
            If the requested plotting library cannot be imported.
        IndexError
            If `mode` ≥ `state.d`.
        """
        # 1) Basic checks and extract ℏ from config:
        ℏ = float(self._config.hbar)  # e.g. ℏ = 2.0 in your example

        d = self.d
        if not (0 <= mode < d):
            raise IndexError(f"Requested mode {mode} but state has d={d} modes.")

        # 2) Get the single-mode density matrix and its populations ⟨n|ρ|n⟩ for variance estimates:
        #    PureFockState.density_matrix is shape (c^d × c^d), but we want to reduce to 'mode'.
        #    Conveniently, PureFockState.reduced((mode,)) returns a *mixed* FockState,
        #    but we can still get ⟨x^2⟩ and ⟨p^2⟩ via that mixed state's density matrix.
    

        # Convert to a mixed FockState on just {mode} by calling .reduced((mode,)):
        mixed_single_mode: FockState = self.reduced((mode,))
        dm: np.ndarray = mixed_single_mode.density_matrix
        # dm has shape (c, c), where c = cutoff.  We can compute ⟨x⟩, ⟨x²⟩, ⟨p²⟩ via matrix elements:
        # Build ⟨m| x |n⟩ and ⟨m| p |n⟩ in the Fock basis truncated at c
        cutoff = self._config.cutoff
        n_levels = cutoff

        # Create annihilation (a) and creation (a†) in the truncated basis:
        np_mod = self._connector.np
        levels = np_mod.arange(n_levels)
        a_data = np_mod.diag(np_mod.sqrt(levels[1:]), k=-1)  # a_{n, n+1} = √(n+1)
        a = a_data
        adag = a_data.T

        # Quadratures:
        x_op = (adag + a) * np_mod.sqrt(ℏ / 2)   # x = √(ℏ/2)(a + a†)
        p_op = (adag - a) * ( -1j * np_mod.sqrt(ℏ / 2) )  # p = -i√(ℏ/2)(a - a†)

        # Compute means and variances:
        mean_x = np_mod.real(np_mod.trace(dm @ x_op))
        mean_x2 = np_mod.real(np_mod.trace(dm @ (x_op @ x_op)))
        var_x = mean_x2 - mean_x**2

        mean_p = np_mod.real(np_mod.trace(dm @ p_op))
        mean_p2 = np_mod.real(np_mod.trace(dm @ (p_op @ p_op)))
        var_p = mean_p2 - mean_p**2

        sigma_x = np_mod.sqrt(var_x)
        sigma_p = np_mod.sqrt(var_p)

        # 3) Decide grid extents in x and p if not given:
        if xlims is None:
            x_center = mean_x
            x_min = x_center - 4 * sigma_x
            x_max = x_center + 4 * sigma_x
        else:
            x_min, x_max = xlims

        if plims is None:
            p_center = mean_p
            p_min = p_center - 4 * sigma_p
            p_max = p_center + 4 * sigma_p
        else:
            p_min, p_max = plims

        xs = np_mod.linspace(x_min, x_max, resolution)
        ps = np_mod.linspace(p_min, p_max, resolution)
        X, P = np_mod.meshgrid(xs, ps, indexing="xy")  # each shape = (resolution, resolution)

        # 4) Build the y‐integration grid:
        y_range = float(y_integration_range)
        Ys = np_mod.linspace(-y_range, +y_range, y_points)  # for ∫ dy …

        # 5) Precompute the single‐mode wavefunctions 

        # To evaluate Hermite H_n(u) quickly, we'll use numpy.polynomial.hermite.hermval:
        

        def hermite_phys(n: int, u: np.ndarray) -> np.ndarray:
            """
            Evaluate the physicist's Hermite polynomial H_n(u) at array u.
            numpy.polynomial.hermite.hermval takes coeffs for the *probabilist's* Hermite,
            but if we pass an array of length n+1: [0,0,…,0,1] (1 at index n),
            hermval returns H_n(u) (physicist's) by default, because numpy's Hermite basis
            is "physicist's".  In other words, coeffs = [0]*n + [1] ⇒ H_n(u).
            """
            coeffs = np_mod.zeros(n + 1, dtype=float)
            coeffs[-1] = 1.0
            return hermval(u, coeffs)

        # Normalization prefactor: φ_n(q) = (πℏ)^(-1/4) (1/√(2^n n!)) H_n(q/√ℏ) exp(−q²/(2ℏ))
        norm_const = (np_mod.pi * ℏ) ** (-0.25)

        # Precompute √(2^n n!) once:
        # We can do a small table for factorials up to c−1:
        fact = np_mod.cumprod(np_mod.concatenate(([1.0], np_mod.arange(1.0, n_levels))))  # length = c
        two_pow = 2 ** levels  # [2^0, 2^1, …, 2^{c−1}]

        sqrt_denoms = np_mod.sqrt(two_pow * fact)  # length c

        # We'll build φ_n at any q via a helper function, but to vectorize,
        # we create arrays for all (x_i − y_k) points and all (x_i + y_k) points:
        XX = X[..., None]  # shape (res, res, 1)
        PP = P[..., None]  # not used directly here

        # For each y_k in Ys, we want arrays of shape (res, res) for (x − y_k) and (x + y_k):
        #   grids_minus[k] = X − Ys[k]
        #   grids_plus[k]  = X + Ys[k]
        grids_minus = XX - Ys[None, None, :]  # shape = (res, res, y_points)
        grids_plus  = XX + Ys[None, None, :]  # same shape

        # Now build φ_n(grids_minus) and φ_n(grids_plus).  We'll allocate:
        #   phi_minus[n, i, j, k] = φ_n( x_i − y_k )
        #   phi_plus [n, i, j, k] = φ_n( x_i + y_k )
        phi_minus = np_mod.empty((n_levels, resolution, resolution, y_points), dtype=float)
        phi_plus  = np_mod.empty_like(phi_minus)

        for n in range(n_levels):
            # Evaluate H_n at (q / √ℏ); i.e. physically: hermite_phys(n, (x±y)/√ℏ)
            arg_minus = (grids_minus / np_mod.sqrt(ℏ))
            Hn_minus = hermite_phys(n, arg_minus)  # shape (res, res, y_points)

            arg_plus  = (grids_plus / np_mod.sqrt(ℏ))
            Hn_plus  = hermite_phys(n, arg_plus)

            # φ_n(q) = (πℏ)^(-1/4) (1/√(2^n n!)) H_n(q/√ℏ) exp(−q²/(2ℏ))
            gaussian_factor_minus = np_mod.exp(- (grids_minus ** 2) / (2 * ℏ))
            gaussian_factor_plus  = np_mod.exp(- (grids_plus  ** 2) / (2 * ℏ))

            phi_minus[n] = norm_const * (Hn_minus / sqrt_denoms[n]) * gaussian_factor_minus
            phi_plus[n]  = norm_const * (Hn_plus  / sqrt_denoms[n]) * gaussian_factor_plus

        # 6) Build ψ(x−y) and ψ(x+y).  
        full_vec = self.state_vector 
        
        # Implementation:
        occupation_space = self._space  

        # Initialize single‐mode amplitudes c_n = 0 + 0j
        c_n = np_mod.zeros(n_levels, dtype=full_vec.dtype)

        for basis_index in range(occupation_space.shape[0]):
            occs = occupation_space[basis_index]  # e.g. [n_0, n_1, …, n_{d−1}]
            photon_in_chosen_mode = int(occs[mode])
            if photon_in_chosen_mode < n_levels:
                c_n[photon_in_chosen_mode] += full_vec[basis_index]

        # Now c_n holds the (possibly unnormalized) amplitude for mode=|n>.  For a pure state,
        # ∑ |c_n|^2 should be 1 (if state normalized).  We'll renormalize anyway:
        norm = np_mod.sqrt(np_mod.sum(np_mod.abs(c_n) ** 2))
        if norm > 0:
            c_n = c_n / norm

        # 7) For each grid‐point (x_i, p_j), do the †y†‐integral:

        # Let's vectorize over i,j,k as much as possible:
        psi_minus = np_mod.tensordot(c_n, phi_minus, axes=(0, 0))  
        # shape = (res, res, y_points), since c_n⋅φ_n over n⇒ collapse dimension n
        psi_plus  = np_mod.tensordot(c_n, phi_plus,  axes=(0, 0))  
        # same shape

        # Build the factor exp(2 i p y / ℏ) for each (p_j, y_k):
        # P has shape (res, res). We just need the row index: j → p_j, but we'll broadcast:
        # Let `Pij = P[i,j]`. Then for each k: phase[i,j,k] = exp(2 i * Pij * Ys[k] / ℏ).
        P_expanded = P[..., None]  # shape = (res, res, 1)
        Ys_expanded = Ys[None, None, :]  # shape = (1, 1, y_points)
        phase_factor = np_mod.exp((2j * P_expanded * Ys_expanded) / ℏ)  

        integrand = phase_factor * psi_minus * np_mod.conj(psi_plus)  

        # Do the trapezoidal rule in y:
        # trapz over Ys dimension (axis=2):
        W_vals = (1.0 / (np_mod.pi * ℏ)) * np_mod.trapezoid(integrand, Ys, axis=2)  
        # shape = (res, res), but generally complex—take real part (Wigner is real).
        W = np_mod.real(W_vals)

        # 8) Plot using the requested library:
        if library.lower() == "matplotlib":
            try:
                import matplotlib.pyplot as plt
            except ImportError as e:
                raise ImportError(
                    "Matplotlib is required to plot the Fock‐basis Wigner function. "
                    "Install it with, e.g. `pip install matplotlib`."
                ) from e

            fig, ax = plt.subplots(figsize=(6, 5))
            # Use contourf with a symmetric colormap about zero:
            vmax = np_mod.max(np_mod.abs(W))
            im = ax.contourf(
                xs,
                ps,
                W,
                levels=100,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=+vmax,
            )
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$p$")
            ax.set_title(f"Wigner function (PureFockState, mode {mode})")
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label(r"$W(x,p)$")
            plt.tight_layout()
            return fig, ax

        elif library.lower() == "plotly":
            try:
                import plotly.graph_objects as go
            except ImportError as e:
                raise ImportError(
                    "Plotly is required to plot the Fock‐basis Wigner function with `library='plotly'`. "
                    "Install it with, e.g. `pip install plotly`."
                ) from e

            vmax = np_mod.max(np_mod.abs(W))
            fig = go.Figure(
                data=go.Contour(
                    x=xs.tolist(),
                    y=ps.tolist(),
                    z=W.tolist(),
                    colorscale="RdBu",
                    contours=dict(start=-vmax, end=+vmax, size=0.01),
                    reversescale=True,
                )
            )
            fig.update_layout(
                title=f"Wigner function (PureFockState, mode {mode})",
                xaxis_title="x",
                yaxis_title="p",
            )
            return fig

        else:
            raise ValueError(
                f"Unknown library '{library}'. Choose either 'matplotlib' or 'plotly'."
            )
