# Changelog

## [7.1.0] - 2025-12-12

### Added

- `Result.outcome_map`, which returns a mapping from measurement outcomes to
  their frequencies and states.
- Extended gate support in `piquasso.dual_rail_encoding`.
- Support for `Vacuum` and `Create` in `SamplingSimulator` (previously one could
  only use `StateVector` for preparing initial states).
- Support for using expressions instead of lambda functions to specify conditions
  and runtime resolution of instruction parameters. Example:
  ```py
  import numpy as np
  import piquasso as pq

  # Create a program with explicit instruction list.
  program = pq.Program(
      instructions=[
          # Prepare a superposition of three two-mode Fock states:
          # |0,2>, |1,1>, and |2,0>, each with equal amplitude.
          pq.StateVector([0, 2]) * np.sqrt(1 / 3),
          pq.StateVector([1, 1]) * np.sqrt(1 / 3),
          pq.StateVector([2, 0]) * np.sqrt(1 / 3),
          # Measure photon number on mode 1 (mid-circuit measurement).
          pq.ParticleNumberMeasurement().on_modes(1),
          # Apply an unresolved squeezing gate on mode 0.
          # The squeezing parameter r is determined *at runtime*
          # from the measurement outcome (x[-1] = last outcome value).
          pq.Squeezing(r="0.1 * x[-1] ** 2").on_modes(0),
      ]
  )

  # Run the program on a 2-mode Fock simulator for 10 random measurement shots.
  result = pq.PureFockSimulator(d=2).execute(program, shots=10)
  ```
- Support for exporting blackbird code through `Program.to_blackbird_code` and
  `Program.save_as_blackbird_code`.
- More extensive support for postselecting photon numbers in `SamplingSimulator`.

### Fixed

- Inferring the number of modes from the instructions was faulty in the previous
  version (where it was introduced), but has been fixed in this version.
- How the Phaseshift gate is converted from Qiskit in the dual-rail encoding module.

### Minor changes

- Instruction validation is moved to happen runtime, i.e., at `Simulator.execute`
  or `Simulator.execute_instructions`. With this, a more comprehensive
  differentiability support can be enabled. Also, it is not required to set
  `validate=False` in `Config` when the user wants to JIT-compile Piquasso code
  with JAX.
- In the `dual_rail_encoding` module, the representation of a qubit is changed.
  In the current version, a qubit $\ket{0}_{\text{qubit}}$ state is represented by
  $\ket{1,0}_{\text{qumode}}$, whereas previously it was $\ket{0,1}_{\text{qumode}}$.


## [7.0.0] - 2025-11-07

### Added

- `covariance_matrix` property for `fermionic.PureFockState`.
- Validation for input occupation numbers in `StateVector`.
- Tutorial for finding dense k-subgraphs using Gaussian Boson Sampling.
- Support for mid-circuit measurements in a Piquasso program. Simple example:
  ```py
  import numpy as np
  import piquasso as pq

  with pq.Program() as program:
      # Initialize all modes in the vacuum state.
      pq.Q() | pq.Vacuum()

      # Apply a two-mode squeezing gate between modes 0 and 1.
      # The squeezing parameter r = log(1 + sqrt(2)) maximizes the probability of
      # detecting 1.
      pq.Q(0, 1) | pq.Squeezing2(r=np.log(1 + np.sqrt(2)))

      # Postselect on detecting exactly one photon in mode 1.
      # This heralds a single-photon–like non-Gaussian state on mode 0.
      pq.Q(1) | pq.PostSelectPhotons((1,))

      # Mix mode 0 (heralded state) and mode 2 (vacuum) on a 50:50 beamsplitter.
      pq.Q(0, 2) | pq.Beamsplitter5050()

  # Run the program on a 3-mode Fock simulator (cutoff auto-defaults, can be set via Config).
  result = pq.PureFockSimulator(d=3).execute(program)
  ```
- Conditional instruction execution via the `Instruction.when(<condition>)` method.
  Simple example:
  ```py
  import numpy as np
  import piquasso as pq

  with pq.Program() as program:
      # Start in |1,0,0>
      pq.Q(0, 1, 2) | pq.StateVector((1, 0, 0))

      # Split the photon between modes 0 and 1
      pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi/4)

      # Mid-circuit photon detection on mode 0
      pq.Q(0) | pq.ParticleNumberMeasurement()

      # Adaptive router: if we detected 0 on mode 0, swap mode 1 to mode 2
      pq.Q(1, 2) | pq.Beamsplitter(theta=np.pi/2).when(
          lambda outcomes: outcomes[-1] == 0
      )

      # Final photon detection on remaining modes
      pq.Q(1, 2) | pq.ParticleNumberMeasurement()

  res = pq.PureFockSimulator(d=3).execute(program, shots=20)
  ```
- Automatic inference of number of modes. Example:
  ```py
  import piquasso as pq

  with pq.Program() as program:
      pq.Q(0, 1) | pq.StateVector((2, 0))
      pq.Q(0, 1) | pq.Beamsplitter5050()

  simulator = pq.PureFockSimulator()  # Number of modes not specified!

  result = simulator.execute(program)
  ```
- Acquire exact probability distributions with the `shots=None` setting.
- Runtime resolution of instruction parameters, as demonstrated by the following
  simple example:
  ```py
  import numpy as np
  import piquasso as pq

  # Create a program with explicit instruction list.
  program = pq.Program(
      instructions=[
          # Prepare a superposition of three two-mode Fock states:
          # |0,2>, |1,1>, and |2,0>, each with equal amplitude.
          pq.StateVector([0, 2]) * np.sqrt(1 / 3),
          pq.StateVector([1, 1]) * np.sqrt(1 / 3),
          pq.StateVector([2, 0]) * np.sqrt(1 / 3),
          # Measure photon number on mode 1 (mid-circuit measurement).
          pq.ParticleNumberMeasurement().on_modes(1),
          # Apply an unresolved squeezing gate on mode 0.
          # The squeezing parameter r is determined *at runtime*
          # from the measurement outcome (x[-1] = last outcome value).
          pq.Squeezing(r=lambda x: 0.1 * x[-1] ** 2).on_modes(0),
      ]
  )

  # Run the program on a 2-mode Fock simulator for 10 random measurement shots.
  result = pq.PureFockSimulator(d=2).execute(program, shots=10)
  ```
- `piquasso.dual_rail_encoding` module. Simple example usage:
  ```py
  from piquasso.dual_rail_encoding import dual_rail_encode_from_qiskit
  from qiskit import QuantumCircuit
  import piquasso as pq
  import numpy as np

  qc = QuantumCircuit(2, 2)
  qc.h(0)
  qc.h(1)
  qc.cz(0, 1)
  qc.measure([0, 1], [0, 1])

  simulator = pq.PureFockSimulator(config=pq.Config(cutoff=8))

  program = dual_rail_encode_from_qiskit(qc)
  result = simulator.execute(program, shots=1000)
  ```

### Fixed

- `sqrtm` changed behavior in SciPy version 1.16, which cause the Takagi decomposition
  to fail in rare cases, therefore, the algorithm got rewritten to be more robust.

### Breaking changes

- Remove `postselect_modes` input argument for `PostSelectPhotons`, `pq.Q`/`on_modes`
  to be used instead.
- `ParticleNumberMeasurement` now returns the unmeasured reduced state of the
  post-measurement state in the `Result` object.
- The following measurements no longer return a state object in the `Result` object
  (corresponding to the pre-measurement state), but `None` instead:
  - `HomodyneMeasurement` on `PureFockSimulator`;
  - `ParticleNumberMeasurement`, `ThresholdMeasurement` on `GaussianSimulator`;
  - `ParticleNumberMeasurement` on `SamplingSimulator`.
- A calculation function (which has to be registered in `_instruction_map`) now needs
  to return with a list of `Branch` objects, instead of a single `Result` object. This
  is only relevant for a user using the API directly.
- During execution, the list of instructions were modified previously for resolving
  `modes`, when it is not specified. With this change, the `modes` parameter for each
  instruction is left untouched.
- The list of samples got a fixed type `List[Tuple[Union[int, float],...]]` (previously
  one could get numpy arrays as well).


## [6.2.0] - 2025-09-03

### Added

- `Result.get_counts` method for showing repetitions in the obtained samples.
- `dask` support for Gaussian Boson Sampling.
- State vector display with occupation numbers using `fock_amplitude_map`.

### Fixed

- Handling number of modes mismatch in `Simulator.execute`.
- Typos in the documentation corrected.
- Instruction mode validation improved for caching invalid indices.
- Jupyter notebook download issue in the documentation.
- Unnecessary singularity check deleted in `Graph`.


## [6.1.0] - 2025-06-23

### Added

- Fermionic `GaussianHamiltonian` support for subsystem.
- The analog of the `Squeezing2` gate in the fermionic setting.
- Fermionic `ControlledPhase` gate.
- Fermionic Ising XX coupling gate.
- Fermionic `PureFockState.fock_probabilities_map`.
- `get_marginal_fock_probabilities` methods for `GaussianState`, `FockState`
  and `PureFockState` for calculating the marginal particle detection
  probabilities.
- The `SamplingState.get_particle_detection_probability` method is made
  differentiable via JAX.
- Support for nested program definitions.
- `GaussianState.get_xp_string_moment` for calculating higher XP-string moments
  for Gaussian states.
- `fock_amplitudes_map` method for `FockState`, `PureFockState` and
  `BatchPureFockState`, a defaultdict containing the probability amplitudes,
  indexed by tuples of occupation numbers on each mode.
- Support for Python 3.13.
- `plot_wigner` method for `FockState`, `PureFockState` and `GaussianState`,
  which can be used to visualize the marginal Wigner function on a single mode.

### Fixed

- JAX has been mistakenly loaded when importing Piquasso, but has been fixed in
  this release.
- Numba reflected list warning got removed.
- Some typos in the documentation got fixed.
- Pfaffian, torontonian and loop torontonian precision type is fixed.

### Performance improvements

- Pfaffian got reimplemented in C++ using the Parlett-Reid algorithm.
- `SLOS_full` algorithm implemented for state vector calculation in `SamplingState`.
- Permanent function got reimplemented in C++ using the algorithm described in
  https://arxiv.org/abs/2309.07027.


## [6.0.0] - 2025-03-26

### Added

- Minimal `fermionic.PureFockSimulator` implementation.
- `GaussianState.get_purity()` function for calculating Gaussian state purity.

### Fixed

- Passive linear operations in `fermionic.GaussianSimulator` time evolution direction.

### Breaking changes

- Majorana operator ordering changed in `piquasso.fermionic` package.
- Phaseshifters are commuted to the end of the circuit in the Clements decomposition.

## [5.0.1] - 2024-10-06

### Fixed

- Windows binary wheel packaging.


## [5.0.0] - 2024-10-01

### Added

- `get_purity` in `*FockState` for calculating the purity.
- `PostSelectPhotons` and `ImperfectPostSelectPhotons` for `PureFockSimulator` and
  `SamplingSimulator`.
- `Beamsplitter5050` support.
- `SamplingState.state_vector`, analog of `*FockState.state_vector`.
- JAX support for `SamplingSimulator`.
- Support for multiple occupation numbers in `SamplingSimulator`.
- `PureFockState.get_particle_detection_probability_on_modes`, which is similar to
  `get_particle_detection_probability`, but for the specified modes.
- `HomodyneMeasurement` support for `PureFockSimulator`.
- `cvqnn.get_cvqnn_weight_indices` is created, which enables slicing of the weights when
   needed.
- `PureFockState.variance_photon_number` for calculating the variance of the photon
  number operator.
- Partial JAX support for `GaussianSimulator`.
- `GaussianState.get_threshold_detection_probability`.
- Python 3.12 support.
- Support for `ParticleNumberMeasurement` in `GaussianSimulator` with the config
  `use_torontonian=True` and displaced Gaussian states.
- `Config.validate` flag. If set to `validate=False`, validations are skipped, possibly
  enabling minor speed-up or JIT compilation.
- `piquasso.fermionic` package with support for fermionic Gaussian states.
- Support for differentiable `GaussianState.get_particle_detection_probability`.

### Fixed

- `Simulator.execute` with `initial_state` specified while using `tf.function`.
- `GaussianSimulator` random number generation from `Config.rng`.
- Error message formatting in `Simulator`.
- `Beamsplitter` default parameters.
- `fock_probabilities` differentiability in `PureFockSimulator`.
- `SamplingState.fock_probabilities` returns with probabilities corresponding to all
  particle number sectors.

### Breaking changes

- Delete unused attributes in `SamplingState`.
- Clements decomposition rewritten.
- The original RNG is kept when a `Config` is copied. This is done to prevent unexpected
  behaviour with seeded calculations.
- The config variable `Config.normalize` is deleted. For the same result, one can call
  `State.normalize` at the end of the calculations.
- `Simulator._default_calculator_class` initial value got deleted to avoid confusion.
- Renamed `Calculator` to `Connector` and corresponding names containing the term
  `calculator`, referring to the original `Calculator` class (e.g.,
  `_default_calculator_class` -> `_default_connector_class`).
- `BaseConnector` (former `BaseCalculator`) rewritten as an abstract class.
- Added/updated `__repr__` methods for all classes in the Piquasso API.

### Performance improvements

- Hafnian, loop hafnian, torontonian and permanent implementations replaced with faster
  implementations. The torontonian calculation is written in C++, and is distributed in
  the wheels alongside the Python code.
- Faster `FockState.norm`.
- Faster `SamplingState.get_particle_detection_probability`.
- JIT compilation of passive linear gates in `*FockSimulator`.
- Common Fock-space related calculations got rewritten, JIT compilation enabled.
- More efficient sampling algorithms for BS and GBS simulations.


## [4.0.0] - 2024-02-29

### Added

- Python 3.11 support.
- Purification of Gaussian states.
- `PureFockState.get_tensor_representation` for embedding the state vector into
  a tensor with rank equal to the number of modes.
- Batch processing of pure Fock states.
- CVQNN module.
- Support for `tf.function` in `PureFockSimulator`.
- Supporting JAX in `PureFockSimulator`.

### Fixed

- Error in custom gradient of passive linear gates in `PureFockSimulator`.

### Breaking changes

- Python 3.7 support dropped.
- `TensorflowPureFockSimulator` has been deleted. Instead, one can use
  `PureFockSimulator` with `TensorflowConnector` specified.
- Printing format of Fock states have been changed.
- Renamed `_state_vector` to `state_vector`.


## [3.0.0] - 2023-10-23

### Added

- `dtype` configuration variable in `Config` to set precision of calculations.
- `Kerr` gate calculation in `PureFockSimulator` performance improved.
- Gradient calculation in `TensorflowPureFockSimulator` got improved.
- A method called `PureFockState.mean_position` which calculates the average position
  in a specified mode.
- `normalize` configuration variable in `Config` to enable/disable normalization of
  states `PureFockSimulator` and `FockSimulator`.

### Changed

- `Displacement` parametrization is simplified.
- The autoscaling of single mode gates got deleted.

### Fixed

- Vacuum state typing issue fixed in `PureFockSimulator`.
- Calculation error during gradient calculation.


## [2.2.0] - 2023-03-07

### Changed

- Several major performance improvements got implemented in
  `TensorflowPureFockSimulator`.

### Fixed

- `quantum-blackbird` version got bumped for `numpy` compatibility.
- `Kerr` gate was applied with a wrong equation, it got corrected.
- Typing error was fixed when applying `Interferometer` in
  `TensorflowPureFockSimulator`.


## [2.1.0] - 2023-02-08

### Added

- Performance increase for the `PureFockSimulator` and `TensorflowPureFockSimulator`.

### Fixed

- `GaussianState.fidelity` gave incorrect results for multiple modes and it
  needed to be corrected.
- During Williamson decomposition, sometimes `scipy.linalg.sqrtm` returned with
  complex matrices instead of real ones which caused problems so it is manually
  casted to real.
- In `TensorflowPureFockSimulator`, the gradient of the displacement gate
  matrix was not applied properly to the upstream gradient, a conjugation is
  added.
- Using `TensorflowPureFockSimulator`, the input of the `Interferometer` gate
  was not converted to a `tensorflow.Tensor` automatically, which has been
  included.


## [2.0.0] - 2022-10-30

### Added

- A simulator class called `TensorflowPureFockSimulator`, which uses Tensorflow
  and is able to calculate gradients.

### Changed

-  By enabling Tensorflow support, we dropped support for customizing the
   permanent, hafnian and loop hafnian calculations through `Config`.


## [1.0.1] - 2022-09-05

### Fixed

- `SamplingSimulator` along with `Loss` using uniform transmissivity
  (transmittance) parameters and `ParticleNumberMeasurement` produced only-zero
  samples instead of the expected samples.


## [1.0.0] - 2022-04-26

### Added

- Support for `Attenuator` in `FockSimulator` and `PureFockSimulator`.
- `LossyInterferometer` channel for `SamplingSimulator`.
- Possibility to provide the `hafnian` as a function in `piquasso.Config`.

### Changed

- The occupation number order has been changed from lexicographic to
  anti-lexicographic.
- The `piquasso.api.errors` module got renamed to `piquasso.api.exceptions`.
- The `Sampling` measurement class has been unified with
  `ParticleNumberMeasurement`.

### Fixed

- `Kerr` gate calculations were incorrect for multiple modes in `FockSimulator`
  and `PureFockSimulator`.
- Calculations that were incorrect for multiple modes using `FockSimulator`
  along with `Squeezing`, `Displacement` or `CubicPhase`.
- Symmetric tensorpower ordering in Fock-backend simulations.
- Handling 0 matrix elements at Clements decomposition.

