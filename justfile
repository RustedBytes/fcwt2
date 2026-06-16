format:
    cargo fmt

test:
    cargo test

lint:
    cargo clippy --all-targets

bench:
    cargo bench --bench cwt

python-build:
    cargo build --features python
