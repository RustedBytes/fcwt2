# fCWT in Rust

[![CI](https://github.com/RustedBytes/fcwt2/actions/workflows/ci.yml/badge.svg)](https://github.com/RustedBytes/fcwt2/actions/workflows/ci.yml)
[![Crates.io Version](https://img.shields.io/crates/v/fcwt2)](https://crates.io/crates/fcwt2)
[![PyPI - Version](https://img.shields.io/pypi/v/fcwt2)](https://pypi.org/project/fcwt2/)

Rust implementation of the fast Continuous Wavelet Transform.

It uses [`rustfft`](https://crates.io/crates/rustfft) for FFT planning and
execution, and nightly Rust `portable_simd` for the daughter-wavelet frequency
multiplication.

## Toolchain

The crate is pinned to nightly in `rust-toolchain.toml` because
`std::simd`/`portable_simd` is still unstable.

```sh
cargo test
cargo clippy --all-targets
cargo fmt --check
cargo bench --bench cwt
```

If you want to call the toolchain explicitly:

```sh
cargo +nightly-2026-04-03 test
```

## Usage

```rust
use fcwt2::{Fcwt, Morlet, ScaleType, Scales};

let signal = vec![0.0_f32; 1024];
let scales = Scales::new(ScaleType::LinearFrequencies, 1_000, 1.0, 100.0, 64)?;
let mut fcwt = Fcwt::new(Morlet::new(2.0));

let coefficients = fcwt.cwt_real(&signal, &scales);
assert_eq!(coefficients.len(), signal.len() * scales.len());
# Ok::<(), fcwt2::ScaleError>(())
```

Output is scale-major: `coefficients[scale_index * signal.len() + sample_index]`.

## Benchmarks

Criterion benchmarks cover real and complex CWT transforms at 1024 and 4096
samples:

```sh
cargo bench --bench cwt
```

## Examples

Run the wavelet packet denoising example to see article-style basis selection
combined with soft-threshold reconstruction:

```sh
cargo run --example wavelet_packet_denoiser
```

### My homelab benchmarks

```
Running benches/cwt.rs (target/release/deps/cwt-6c2d4260722aeb78)
cwt_real/1024           time:   [320.71 µs 321.58 µs 322.68 µs]
                        thrpt:  [203.10 Melem/s 203.79 Melem/s 204.34 Melem/s]
Found 3 outliers among 100 measurements (3.00%)
  3 (3.00%) high severe
Benchmarking cwt_real/4096: Warming up for 3.0000 s
cwt_real/4096           time:   [1.6010 ms 1.6288 ms 1.6622 ms]
                        thrpt:  [157.71 Melem/s 160.95 Melem/s 163.74 Melem/s]
Found 15 outliers among 100 measurements (15.00%)
  2 (2.00%) high mild
  13 (13.00%) high severe

cwt_complex/1024        time:   [325.21 µs 326.07 µs 327.12 µs]
                        thrpt:  [200.34 Melem/s 200.99 Melem/s 201.52 Melem/s]
Found 3 outliers among 100 measurements (3.00%)
  2 (2.00%) high mild
  1 (1.00%) high severe
Benchmarking cwt_complex/4096: Warming up for 3.0000 s
cwt_complex/4096        time:   [1.4950 ms 1.4991 ms 1.5039 ms]
                        thrpt:  [174.31 Melem/s 174.86 Melem/s 175.35 Melem/s]
Found 1 outliers among 100 measurements (1.00%)
  1 (1.00%) high mild
```

## Python bindings

PyO3 bindings are available behind the optional `python` feature:

```sh
cargo build --features python
```

The module exposes `Morlet`, `Scales`, `Fcwt`/`FCWT`,
`WaveletPacketTransform`, `StationaryWaveletTransform`, and
`DualTreeComplexWaveletTransform`. Real transforms accept a list of floats;
complex transforms accept Python `complex` values or `(real, imag)` tuples and
return Python `complex` values in the same scale-major layout as the Rust API.

To build/install the Python extension with maturin:

```sh
pip install maturin
maturin develop
```

Example:

```python
import fcwt2

scales = fcwt2.Scales.linear_frequencies(1000, 1.0, 120.0, 64)
transform = fcwt2.Fcwt(2.0)
coefficients = transform.cwt_real([0.0] * 1024, scales)

complex_coefficients = transform.cwt_complex([1.0 + 0.0j] * 1024, scales)

packet = fcwt2.WaveletPacketTransform(3)
tree = packet.decompose([0.0] * 8)
reconstructed = tree.reconstruct()

swt = fcwt2.StationaryWaveletTransform(2)
swt_coefficients = swt.decompose([0.0] * 8)

dtcwt = fcwt2.DualTreeComplexWaveletTransform(2)
dtcwt_tree = dtcwt.decompose([0.0] * 8)
```

Tagged releases matching `v*` build Python wheels on Linux, macOS, and Windows
with GitHub Actions and upload the wheels plus source distribution to the
corresponding GitHub Release.

## Acknowledgements

- Original library fCWT: https://github.com/fastlib/fCWT
- Fixed fCWT in C++: https://github.com/ThirdLetterC/fCWT 
- PyWavelets - Wavelet Transforms in Python: https://pywavelets.readthedocs.io/en/latest/
- Wavelet Basis Selection in Signal Denoising Based on Wavelet-Coefficient Distribution Shape: https://www.mdpi.com/2624-6120/7/3/39
- dtcwt library: https://dtcwt.readthedocs.io/en/0.12.0/
