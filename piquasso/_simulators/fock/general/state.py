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

from typing import Optional, Tuple, Any, Generator, Dict, Literal

import numpy as np
from numpy.polynomial.hermite import hermval
from piquasso.api.connector import BaseConnector

from piquasso.api.config import Config
from piquasso.api.exceptions import InvalidState, PiquassoException
from piquasso._math.linalg import is_selfadjoint
from piquasso._math.fock import cutoff_fock_space_dim, get_fock_space_basis
from piquasso._math.indices import (
    get_index_in_fock_space,
    get_index_in_fock_space_array,
)

from ..state import BaseFockState


class FockState(BaseFockState):
    """Object to represent a general bosonic state in the Fock basis.

    Note:
        If you only work with pure states, it is advised to use
        :class:`~piquasso._simulators.fock.pure.state.PureFockState` instead.
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

        self._density_matrix = self._get_empty()

    def _get_empty(self) -> np.ndarray:
        state_vector_size = cutoff_fock_space_dim(cutoff=self._config.cutoff, d=self.d)
        return np.zeros(
            shape=(state_vector_size,) * 2, dtype=self._config.complex_dtype
        )

    def reset(self) -> None:
        self._density_matrix = self._get_empty()
        self._density_matrix[0, 0] = 1.0

    def _as_mixed(self):
        return self.copy()

    @classmethod
    def from_fock_state(cls, state: BaseFockState) -> "FockState":
        """Instantiation using another :class:`BaseFockState` instance.

        Args:
            state (BaseFockState):
                The instance from which a :class:`FockState` instance is created.
        """

        new_instance = cls(d=state.d, connector=state._connector, config=state._config)

        new_instance._density_matrix = state.density_matrix

        return new_instance

    @property
    def nonzero_elements(
        self,
    ) -> Generator[Tuple[complex, Tuple], Any, None]:
        np = self._connector.np

        nonzero_indices = np.nonzero(self._density_matrix)

        row_indices, col_indices = nonzero_indices

        row_occupation_numbers = self._space[row_indices]
        col_occupation_numbers = self._space[col_indices]

        nonzero_elements = self._density_matrix[nonzero_indices]

        for index, coefficient in enumerate(nonzero_elements):
            yield coefficient, (
                tuple(row_occupation_numbers[index]),
                tuple(col_occupation_numbers[index]),
            )

    def __str__(self) -> str:
        return " + ".join(
            [
                str(coefficient) + str(basis)
                for coefficient, basis in self.nonzero_elements
            ]
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FockState):
            return False
        return np.allclose(self._density_matrix, other._density_matrix)

    @property
    def density_matrix(self) -> np.ndarray:
        cardinality = cutoff_fock_space_dim(d=self.d, cutoff=self._config.cutoff)

        return self._density_matrix[:cardinality, :cardinality]

    def get_particle_detection_probability(
        self, occupation_number: np.ndarray
    ) -> float:
        if self._config.validate and len(occupation_number) != self.d:
            raise PiquassoException(
                f"The specified occupation number should have length '{self.d}': "
                f"occupation_number='{occupation_number}'."
            )

        index = get_index_in_fock_space(occupation_number)
        return np.diag(self._density_matrix)[index].real

    @property
    def fock_probabilities(self) -> np.ndarray:
        return self._connector.np.real(self._connector.np.diag(self._density_matrix))

    @property
    def fock_probabilities_map(self) -> Dict[Tuple[int, ...], float]:
        probability_map: Dict[Tuple[int, ...], float] = {}

        for index, basis in enumerate(self._space):
            probability_map[tuple(basis)] = np.abs(self._density_matrix[index, index])

        return probability_map

    def reduced(self, modes: Tuple[int, ...]) -> "FockState":
        np = self._connector.np
        fallback_np = self._connector.fallback_np
        d = self.d

        if modes == tuple(range(d)):
            return self

        outer_modes = self._get_auxiliary_modes(modes)

        inner_size = len(modes)
        outer_size = d - inner_size

        reduced_state = FockState(
            d=inner_size, connector=self._connector, config=self._config
        )

        density_list = [
            [0.0 for _ in range(reduced_state._density_matrix.shape[1])]
            for _ in range(reduced_state._density_matrix.shape[0])
        ]

        cutoff = self._config.cutoff
        inner_space = get_fock_space_basis(d=inner_size, cutoff=cutoff)
        outer_space = get_fock_space_basis(d=outer_size, cutoff=cutoff)

        index_list = []

        basis_matrix = fallback_np.empty(
            shape=(outer_space.shape[0], self.d), dtype=int
        )

        for inner_basis in inner_space:
            size = cutoff_fock_space_dim(
                cutoff=cutoff - fallback_np.sum(inner_basis), d=outer_size
            )

            basis_matrix[:size, modes] = inner_basis

            basis_matrix[:size, outer_modes] = outer_space[:size]

            partial_index_list = get_index_in_fock_space_array(basis_matrix[:size])

            index_list.append(partial_index_list)

        for i, ix1 in enumerate(index_list):
            for j, ix2 in enumerate(index_list):
                minlen = min(len(ix1), len(ix2))
                density_list[i][j] = np.sum(
                    self._density_matrix[ix1[:minlen], ix2[:minlen]]
                )

        reduced_state._density_matrix = np.array(density_list)

        return reduced_state

    @property
    def norm(self) -> float:
        np = self._connector.np
        return np.real(np.trace(self._density_matrix))

    def normalize(self) -> None:
        """Normalizes the density matrix to have a trace of 1.

        Raises:
            RuntimeError: Raised if the current norm of the state is too close to 0.
        """
        norm = self.norm

        if self._config.validate and np.isclose(norm, 0):
            raise InvalidState("The norm of the state is 0.")

        self._density_matrix = self._density_matrix / self.norm

    def validate(self) -> None:
        """Validates the represented state.

        Raises:
            InvalidState:
                Raised, if the density matrix is not positive semidefinite, not
                self-adjoint or the trace of the density matrix is not 1.
        """
        if not self._config.validate:
            return

        if not is_selfadjoint(self._density_matrix):
            raise InvalidState(
                "The density matrix is not self-adjoint:\n"
                f"density_matrix={self._density_matrix}"
            )

        fock_probabilities = self.fock_probabilities

        if not np.all(fock_probabilities >= 0.0):
            raise InvalidState(
                "The density matrix is not positive semidefinite.\n"
                f"fock_probabilities={fock_probabilities}"
            )

        trace_of_density_matrix = sum(fock_probabilities)

        if not np.isclose(trace_of_density_matrix, 1.0):
            raise InvalidState(
                f"The trace of the density matrix is {trace_of_density_matrix}, "
                "instead of 1.0:\n"
                f"fock_probabilities={fock_probabilities}"
            )

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
        reduced_dm = self.reduced(modes=modes).density_matrix
        annih = np.diag(np.sqrt(np.arange(1, self._config.cutoff)), 1)
        create = annih.T
        position = (create + annih) * np.sqrt(self._config.hbar / 2)
        momentum = -1j * (annih - create) * np.sqrt(self._config.hbar / 2)

        if phi != 0:
            rotated_quadratures = position * np.cos(phi) + momentum * np.sin(phi)
        else:
            rotated_quadratures = position

        expectation = np.trace(np.dot(reduced_dm, rotated_quadratures)).real
        variance = (
            np.trace(
                np.dot(reduced_dm, np.dot(rotated_quadratures, rotated_quadratures))
            ).real
            - expectation**2
        )
        return expectation, variance

    def get_purity(self):
        np = self._connector.np
        density_matrix = self.density_matrix
        return np.real(np.einsum("ij,ji", density_matrix, density_matrix))

    def plot_wigner_fockstate(
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
        Plot the single‐mode Wigner function of a (mixed) FockState by numerical integration.

        This uses the formula
            W(x,p) = (1 / (π ℏ)) ∫_{-y_max}^{+y_max} e^{(2 i p y)/ℏ} 
                      ⟨ x − y | ρ | x + y ⟩ dy,
        where ρ is the single‐mode density matrix in Fock basis, and
            ⟨q | n⟩ = φ_n(q) = (π ℏ)^(-1/4) (1/√(2^n n!)) H_n(q/√ℏ) e^(−q^2/(2ℏ)).

        Steps:
          1. Reduce the full `FockState` to obtain the single‐mode density matrix ρ (shape = cutoff × cutoff).
          2. Build a uniform grid in x and p (each axis of length = resolution).
          3. Build a uniform grid of y∈[−y_integration_range, +y_integration_range] (length = y_points).
          4. Precompute φ_n(x_i ± y_k) for n=0..cutoff−1, i=0..resolution−1, k=0..y_points−1.
          5. For each y_k, form the matrix
               M_k[i,j] = sum_{m,n=0}^{cutoff−1}  ρ[m,n] · φ_m(x_i − y_k) · conj[φ_n(x_i + y_k)].
          6. Multiply M_k[i,j] by exp(2 i p_j y_k / ℏ), integrate over y via trapezoidal rule,
             and multiply by (1/(π ℏ)) ⇒ W[i,j].
          7. Defer the import of Matplotlib or Plotly until plotting. If missing, raise ImportError.

        Parameters
        ----------
        mode : int, default=0
            Which mode index (0‐based) to visualize. Must satisfy 0 ≤ mode < state.d.
        xlims : (float, float) or None
            If provided, these are (x_min, x_max). Otherwise, defaults to ±4 σ_x, where
            σ_x = √⟨x²⟩ for that single mode (computed from its reduced density).
        plims : (float, float) or None
            If provided, these are (p_min, p_max). Otherwise, defaults to ±4 σ_p.
        resolution : int, default=200
            Number of grid points along *each* of the x and p axes
            (so the total grid is resolution×resolution).
        y_integration_range : float, default=5.0
            We truncate the y‐integral to [−y_range, +y_range]. If ℏ is small,
            you may need a larger range; if it's big or the state is narrow,
            you can choose smaller.
        y_points : int, default=200
            Number of points to discretize the y‐integral on [−y_range, +y_range].
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

        # 1) Extract ℏ, dimension d, and check `mode`
        ℏ = float(self._config.hbar)
        d = self.d
        if not (0 <= mode < d):
            raise IndexError(f"Requested mode {mode} but state.d = {d}.")

        # 2) Build the single‐mode (cutoff×cutoff) density matrix ρ:
        #    FockState.reduced((mode,)) returns a FockState on only that mode.
        mixed_single = self.reduced((mode,))
        ρ = mixed_single.density_matrix     # shape = (cutoff, cutoff), complex128
        cutoff = ρ.shape[0]

        # 3) From ρ we can compute ⟨x⟩, ⟨x²⟩, ⟨p⟩, ⟨p²⟩ to get variances for default plotting windows.
        #    Build annihilation/creation in truncated basis:
        np_mod = self._connector.np
        levels = np_mod.arange(cutoff)
        a_data = np_mod.diag(np_mod.sqrt(levels[1:]), k=-1)  # a_{n,n+1} = √(n+1)
        a = a_data
        adag = a_data.T
        x_op = (adag + a) * np_mod.sqrt(ℏ / 2)                     # x = √(ℏ/2)(a + a†)
        p_op = (adag - a) * ( -1j * np_mod.sqrt(ℏ / 2) )           # p = -i√(ℏ/2)(a - a†)

        mean_x = np_mod.real(np_mod.trace(ρ @ x_op))
        mean_x2 = np_mod.real(np_mod.trace(ρ @ (x_op @ x_op)))
        var_x = mean_x2 - mean_x**2

        mean_p = np_mod.real(np_mod.trace(ρ @ p_op))
        mean_p2 = np_mod.real(np_mod.trace(ρ @ (p_op @ p_op)))
        var_p = mean_p2 - mean_p**2

        sigma_x = np_mod.sqrt(var_x)
        sigma_p = np_mod.sqrt(var_p)

        # 4) Decide on x‐ and p‐axis extents if not provided:
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

        # 5) Build y‐grid for the integral:
        y_range = float(y_integration_range)
        Ys = np_mod.linspace(-y_range, +y_range, y_points)  # shape = (y_points,)

        # 6) Precompute φ_n(q) for all n=0..cutoff‐1 at  q = X_{ij} ± Ys[k].
        #    We will need φ_n(x_i - y_k) and φ_n(x_i + y_k) for all (i,j,k).
        #    First: build a helper for physicist's Hermite H_n(u):

        def hermite_phys(n: int, u: np.ndarray) -> np.ndarray:
            """
            Evaluate the physicist's Hermite polynomial H_n(u) at array u.
            numpy.polynomial.hermite.hermval with coeffs = [0,...,0,1] gives H_n(u).
            """
            coeffs = np_mod.zeros(n + 1, dtype=float)
            coeffs[-1] = 1.0
            return hermval(u, coeffs)

        #    Normalization prefactor: φ_n(q) = (πℏ)^(-1/4) (1/√(2^n n!)) H_n(q/√ℏ) e^(−q²/(2ℏ))
        norm_prefac = (np_mod.pi * ℏ) ** (-0.25)

        #    Precompute √(2^n n!) for n=0..cutoff−1:
        fact = np_mod.cumprod(np_mod.concatenate(([1.0], np_mod.arange(1.0, cutoff))))  # length=cutoff
        two_pow = 2 ** levels  # [2^0, 2^1, ..., 2^(cutoff−1)]
        sqrt_denoms = np_mod.sqrt(two_pow * fact)  # length = cutoff

        #    We need to build arrays:
        #      phi_minus[n, i, j, k] = φ_n( x_i − y_k )
        #      phi_plus [n, i, j, k] = φ_n( x_i + y_k )
        #
        #    Let N = resolution×resolution.  First, reshape X so we can broadcast with Ys:
        XX = X[..., None]  # shape = (res, res, 1)
        grids_minus = XX - Ys[None, None, :]   # shape = (res, res, y_points)
        grids_plus  = XX + Ys[None, None, :]   # shape = (res, res, y_points)

        #    Allocate storage:
        phi_minus = np_mod.empty((cutoff, resolution, resolution, y_points), dtype=float)
        phi_plus  = np_mod.empty_like(phi_minus)

        for n in range(cutoff):
            # Evaluate H_n at (q/√ℏ):
            arg_minus = (grids_minus / np_mod.sqrt(ℏ))           # shape = (res, res, y_points)
            Hn_minus = hermite_phys(n, arg_minus)                 # shape = (res, res, y_points)

            arg_plus  = (grids_plus / np_mod.sqrt(ℏ))
            Hn_plus  = hermite_phys(n, arg_plus)

            # Gaussian factors:
            gauss_minus = np_mod.exp(- (grids_minus ** 2) / (2 * ℏ))
            gauss_plus  = np_mod.exp(- (grids_plus  ** 2) / (2 * ℏ))

            # φ_n(q) = norm_prefac * H_n(q/√ℏ)/√(2^n n!) * e^(−q²/(2ℏ))
            denom = sqrt_denoms[n]
            phi_minus[n] = norm_prefac * (Hn_minus / denom) * gauss_minus
            phi_plus[n]  = norm_prefac * (Hn_plus  / denom) * gauss_plus

        # 7) For each y_k (k=0..y_points−1), form
        #      M_k[i,j] := sum_{m,n=0..cutoff−1}  ρ[m,n] · φ_m(x_i − y_k) · conj[φ_n(x_i + y_k)].
        #
        #    We will vectorize over (i,j) by flattening to N = res×res.  Then:
        #      phi_minus_k (shape = (cutoff, N)) := φ_m evaluated on all (i,j) at y_k
        #      phi_plus_conj_k (shape = (cutoff, N)) := conj[φ_n(x_i + y_k)] for all (i,j)
        #
        #    Then: D_k = ρ @ phi_plus_conj_k  (shape = (cutoff, N) )
        #          M_k_flat = sum_{m=0}^{cutoff−1}  phi_minus_k[m,ℓ] * D_k[m,ℓ]  for ℓ=0..N−1
        #
        #    Finally M_k = reshape(M_k_flat, (res, res)).
        N = resolution * resolution
        phi_minus_flat  = phi_minus.reshape(cutoff, N, y_points)   # shape = (cutoff, N, y_points)
        phi_plus_flat   = phi_plus.reshape(cutoff, N, y_points)    # shape = (cutoff, N, y_points)
        phi_plus_conj_flat = np_mod.conj(phi_plus_flat)

        # Build a container for M over (i,j,k):
        M = np_mod.empty((resolution, resolution, y_points), dtype=complex)

        # Precompute ρ once:
        ρ_mat = ρ  # shape = (cutoff, cutoff)

        # Now loop over k to build M[..., k]:
        for k in range(y_points):
            phi_minus_k      = phi_minus_flat[..., k]      # shape = (cutoff, N)
            phi_plus_conj_k  = phi_plus_conj_flat[..., k]   # shape = (cutoff, N)

            # D_k = ρ @ phi_plus_conj_k   (shape = (cutoff, N))
            D_k = ρ_mat @ phi_plus_conj_k

            # M_k_flat[ℓ] = sum_{m=0..cutoff−1} phi_minus_k[m,ℓ] * D_k[m,ℓ]
            M_k_flat = np_mod.sum(phi_minus_k * D_k, axis=0)   # shape = (N,)

            M[..., k] = M_k_flat.reshape((resolution, resolution))

        # 8) Build the phase factor exp(2 i p y / ℏ) on the same grid:
        P_expanded = P[..., None]      # shape = (res, res, 1)
        Ys_expanded = Ys[None, None, :]  # shape = (1, 1, y_points)
        phase = np_mod.exp((2j * P_expanded * Ys_expanded) / ℏ)   # shape = (res, res, y_points)

        # 9) Form the integrand = M * phase.  Then integrate over y via trapezoidal rule:
        integrand = M * phase   # shape = (res, res, y_points)
        W_vals = (1.0 / (np_mod.pi * ℏ)) * np_mod.trapezoid(integrand, Ys, axis=2)
        W = np_mod.real(W_vals)   # shape = (res, res)

        # 10) Deferred import / plotting:
        if library.lower() == "matplotlib":
            try:
                import matplotlib.pyplot as plt
            except ImportError as e:
                raise ImportError(
                    "Matplotlib is required to plot the Fock‐basis Wigner function. "
                    "Install it with, e.g., `pip install matplotlib`."
                ) from e

            fig, ax = plt.subplots(figsize=(6, 5))
            vmax = np_mod.max(np_mod.abs(W))
            im = ax.contourf(
                xs,
                ps,
                W,
                levels=100,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$p$")
            ax.set_title(f"Wigner function (FockState, mode {mode})")
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
                    "Install it with, e.g., `pip install plotly`."
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
                title=f"Wigner function (FockState, mode {mode})",
                xaxis_title="x",
                yaxis_title="p",
            )
            return fig

        else:
            raise ValueError(
                f"Unknown library '{library}'. Choose either 'matplotlib' or 'plotly'."
            )
