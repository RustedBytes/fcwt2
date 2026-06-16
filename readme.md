# fCWT in Rust

Rust implementation of the fast Continuous Wavelet Transform originally written in C++.

This crate replaces the original C++/KFR implementation with an idiomatic Rust
API. It uses [`rustfft`](https://crates.io/crates/rustfft) for FFT planning and
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
use fcwt::{Fcwt, Morlet, ScaleType, Scales};

let signal = vec![0.0_f32; 1024];
let scales = Scales::new(ScaleType::LinearFrequencies, 1_000, 1.0, 100.0, 64)?;
let mut fcwt = Fcwt::new(Morlet::new(2.0));

let coefficients = fcwt.cwt_real(&signal, &scales);
assert_eq!(coefficients.len(), signal.len() * scales.len());
# Ok::<(), fcwt::ScaleError>(())
```

Output is scale-major: `coefficients[scale_index * signal.len() + sample_index]`.

## Benchmarks

Criterion benchmarks cover real and complex CWT transforms at 1024 and 4096
samples:

```sh
cargo bench --bench cwt
```

## Python bindings

PyO3 bindings are available behind the optional `python` feature:

```sh
cargo build --features python
```

The module exposes `Morlet`, `Scales`, and `Fcwt`. Real transforms accept a
list of floats; complex transforms accept `(real, imag)` tuples and return
Python `complex` values in the same scale-major layout as the Rust API.

To build/install the Python extension with maturin:

```sh
pip install maturin
maturin develop
```

Example:

```python
import fcwt

scales = fcwt.Scales.linear_frequencies(1000, 1.0, 120.0, 64)
transform = fcwt.Fcwt.morlet(2.0)
coefficients = transform.cwt_real([0.0] * 1024, scales)
```

Tagged releases matching `v*` build Python wheels on Linux, macOS, and Windows
with GitHub Actions and upload the wheels plus source distribution to the
corresponding GitHub Release.

## Acknowledgements

- Original library fCWT: https://github.com/fastlib/fCWT
