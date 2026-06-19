# Changelog

All notable changes to this project will be documented in this file.

## v1.6.0

### Added

- Added public discrete wavelet basis support via `DiscreteWavelet` and `WaveletFilterBank`.
- Added built-in Haar, Daubechies (`db2`, `db4`, `db6`, `db8`), and Symlet (`sym2`, `sym4`, `sym6`, `sym8`) filter banks.
- Added `WaveletPacketTransform::with_wavelet(...)` and `StationaryWaveletTransform::with_wavelet(...)` while keeping `new(levels)` as the Haar default.
- Added custom filter-bank construction and validation for finite, matching-length analysis/synthesis filters.
- Added basis scoring and selection helpers: `score_basis`, `select_basis`, `BasisScore`, `SelectedBasis`, `TransformKind`, and `BasisSelectionCriterion`.
- Added coefficient ownership helpers for safer modified-coefficient reconstruction:
  `WaveletPacketTree::into_leaves`, `WaveletPacketTree::from_leaves`, `WaveletPacketTree::from_leaves_with_filter_bank`,
  `SwtCoefficients::into_levels`, and `SwtCoefficients::from_levels`.
- Added Python bindings for named packet/SWT wavelet selection and basis scoring/selection.
- Added acknowledgement for the MDPI article "Wavelet Basis Selection in Signal Denoising Based on Wavelet-Coefficient Distribution Shape".

### Changed

- Wavelet packet reconstruction now uses the tree's synthesis filters instead of hardcoded Haar filters.
- Stationary wavelet reconstruction now uses the transform's synthesis filters instead of a Haar-only inverse.
- Packet and stationary transforms now retain and expose their active filter bank metadata.

### Fixed

- Preserved Haar-compatible default behavior through `new(levels)` for packet and stationary transforms.

### Tests

- Added round-trip reconstruction coverage for supported non-Haar packet and stationary wavelet transforms.
- Added filter-bank validation tests.
- Added basis selector diagnostics and short-coarsest-detail-band error tests.
