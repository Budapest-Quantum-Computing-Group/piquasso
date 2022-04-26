# Changelog

## [0.8.3] - 2022-04-25

### Fixed

- Calculations that were incorrect for multiple modes using `FockSimulator`
  along with `Squeezing`, `Displacement` or `CubicPhase`.

## [0.8.2] - 2022-04-07

### Fixed

- Ordering of symmetric tensorpower on the 1-particle subspace in `PureFockSimulator` and in `FockSimulator`.
- Handling of zero matrix elements in the Clements decomposition under `piquasso.decompositions.clements`.
