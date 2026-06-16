use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fcwt::{Complex32, Fcwt, Morlet, ScaleType, Scales};
use std::hint::black_box;

fn sine_signal(len: usize, sample_rate: usize, frequency: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * frequency * t).sin()
        })
        .collect()
}

fn analytic_signal(len: usize, sample_rate: usize, frequency: f32) -> Vec<Complex32> {
    (0..len)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let phase = 2.0 * std::f32::consts::PI * frequency * t;
            Complex32::new(phase.cos(), phase.sin())
        })
        .collect()
}

fn bench_cwt_real(c: &mut Criterion) {
    let mut group = c.benchmark_group("cwt_real");
    let sample_rate = 1_000;
    let scales = Scales::new(ScaleType::LinearFrequencies, sample_rate, 1.0, 120.0, 64).unwrap();

    for len in [1024_usize, 4096] {
        let signal = sine_signal(len, sample_rate, 32.0);
        group.throughput(Throughput::Elements((len * scales.len()) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(len), &signal, |b, signal| {
            let mut fcwt = Fcwt::new(Morlet::new(2.0));
            b.iter(|| fcwt.cwt_real(black_box(signal), black_box(&scales)));
        });
    }

    group.finish();
}

fn bench_cwt_complex(c: &mut Criterion) {
    let mut group = c.benchmark_group("cwt_complex");
    let sample_rate = 1_000;
    let scales = Scales::new(ScaleType::LinearFrequencies, sample_rate, 1.0, 120.0, 64).unwrap();

    for len in [1024_usize, 4096] {
        let signal = analytic_signal(len, sample_rate, 32.0);
        group.throughput(Throughput::Elements((len * scales.len()) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(len), &signal, |b, signal| {
            let mut fcwt = Fcwt::new(Morlet::new(2.0));
            b.iter(|| fcwt.cwt_complex(black_box(signal), black_box(&scales)));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_cwt_real, bench_cwt_complex);
criterion_main!(benches);
