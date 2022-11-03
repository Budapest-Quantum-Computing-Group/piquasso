# Changelog


## [2.0.1] - 2022-11-03

### Fixed

- `GaussianState.fidelity` gave incorrect results for multiple modes and it
  needed to be corrected.


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

