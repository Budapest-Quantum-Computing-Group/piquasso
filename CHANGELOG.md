# Changelog

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
- `PureFockState.get_tensor_representation` for embeddng the state vector into
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

