use rustfft::{FftPlanner, num_complex::Complex32};
use std::simd::{
    Simd,
    num::{SimdFloat, SimdUint},
};

use crate::{Scales, Wavelet, next_power_of_two_len};

const LANES: usize = 8;

#[derive(Clone, Debug)]
pub struct Fcwt<W> {
    wavelet: W,
    normalize: bool,
}

impl<W> Fcwt<W> {
    pub fn new(wavelet: W) -> Self {
        Self {
            wavelet,
            normalize: true,
        }
    }

    pub fn with_normalization(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    pub fn wavelet(&self) -> &W {
        &self.wavelet
    }

    pub fn wavelet_mut(&mut self) -> &mut W {
        &mut self.wavelet
    }
}

impl<W: Wavelet> Fcwt<W> {
    pub fn cwt_real(&mut self, input: &[f32], scales: &Scales) -> Vec<Complex32> {
        let complex_input = input
            .iter()
            .copied()
            .map(|value| Complex32::new(value, 0.0))
            .collect::<Vec<_>>();
        self.cwt_inner(&complex_input, scales, true)
    }

    pub fn cwt_complex(&mut self, input: &[Complex32], scales: &Scales) -> Vec<Complex32> {
        self.cwt_inner(input, scales, false)
    }

    fn cwt_inner(
        &mut self,
        input: &[Complex32],
        scales: &Scales,
        mirror_real_spectrum: bool,
    ) -> Vec<Complex32> {
        if input.is_empty() {
            return Vec::new();
        }

        let size = input.len();
        let fft_size = next_power_of_two_len(size);
        let mut planner = FftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(fft_size);
        let inverse = planner.plan_fft_inverse(fft_size);

        let mut input_hat = vec![Complex32::ZERO; fft_size];
        input_hat[..size].copy_from_slice(input);
        forward.process(&mut input_hat);

        if mirror_real_spectrum {
            for i in 1..(fft_size >> 1) {
                input_hat[fft_size - i] = input_hat[i].conj();
            }
        }

        self.wavelet.generate_frequency(fft_size);

        let mut output = vec![Complex32::ZERO; scales.len() * size];
        let mut multiplied = vec![Complex32::ZERO; fft_size];
        let mut inverse_buffer = vec![Complex32::ZERO; fft_size];

        for (scale_index, scale) in scales.as_slice().iter().copied().enumerate() {
            daughter_wavelet_multiply(
                &input_hat,
                &mut multiplied,
                self.wavelet.mother(),
                scale,
                self.wavelet.imag_frequency(),
                self.wavelet.double_sided(),
            );

            inverse_buffer.copy_from_slice(&multiplied);
            inverse.process(&mut inverse_buffer);
            if self.normalize {
                normalize_inverse_fft(&mut inverse_buffer);
            }

            let start = scale_index * size;
            output[start..start + size].copy_from_slice(&inverse_buffer[..size]);
        }

        output
    }
}

fn normalize_inverse_fft(out: &mut [Complex32]) {
    let size = out.len() as f32;
    for value in out {
        *value /= size;
    }
}

fn daughter_wavelet_multiply(
    input: &[Complex32],
    output: &mut [Complex32],
    mother: &[f32],
    scale: f32,
    imaginary: bool,
    double_sided: bool,
) {
    output.fill(Complex32::ZERO);

    let len = input.len();
    let endpoint = ((len as f32 / 2.0).min(len as f32 * 2.0 / scale)) as usize;
    let step = scale / 2.0;
    let maximum = len as f32 - 1.0;
    let imaginary_sign = if imaginary { -1.0 } else { 1.0 };

    let mut q = 0;
    while q + LANES <= endpoint {
        let indices = Simd::<usize, LANES>::from_array(core::array::from_fn(|lane| q + lane));
        let qf = indices.cast::<f32>();
        let mother_indices = (qf * Simd::splat(step))
            .simd_min(Simd::splat(maximum))
            .cast::<usize>();
        let wav = Simd::<f32, LANES>::from_array(core::array::from_fn(|lane| {
            mother[mother_indices[lane]]
        }));

        let re = Simd::<f32, LANES>::from_array(core::array::from_fn(|lane| input[q + lane].re));
        let im = Simd::<f32, LANES>::from_array(core::array::from_fn(|lane| input[q + lane].im));
        let out_re = re * wav;
        let out_im = im * wav * Simd::splat(imaginary_sign);

        for lane in 0..LANES {
            output[q + lane] = Complex32::new(out_re[lane], out_im[lane]);
        }

        q += LANES;
    }

    for q1 in q..endpoint {
        let tmp = maximum.min(step * q1 as f32) as usize;
        output[q1] = Complex32::new(
            input[q1].re * mother[tmp],
            input[q1].im * mother[tmp] * imaginary_sign,
        );
    }

    if double_sided {
        let last = len - 1;
        let mut q = 0;
        while q + LANES <= endpoint {
            let indices = Simd::<usize, LANES>::from_array(core::array::from_fn(|lane| q + lane));
            let qf = indices.cast::<f32>();
            let mother_indices = (qf * Simd::splat(step))
                .simd_min(Simd::splat(maximum))
                .cast::<usize>();
            let wav = Simd::<f32, LANES>::from_array(core::array::from_fn(|lane| {
                mother[mother_indices[lane]]
            }));

            let source_indices = core::array::from_fn::<_, LANES, _>(|lane| last - q - lane);
            let re = Simd::<f32, LANES>::from_array(source_indices.map(|idx| input[idx].re));
            let im = Simd::<f32, LANES>::from_array(source_indices.map(|idx| input[idx].im));
            let out_re = re * wav * Simd::splat(imaginary_sign);
            let out_im = im * wav;

            for lane in 0..LANES {
                output[source_indices[lane]] = Complex32::new(out_re[lane], out_im[lane]);
            }

            q += LANES;
        }

        for q1 in q..endpoint {
            let tmp = maximum.min(step * q1 as f32) as usize;
            let dest = last - q1;
            output[dest] = Complex32::new(
                input[dest].re * mother[tmp] * imaginary_sign,
                input[dest].im * mother[tmp],
            );
        }
    }
}

#[allow(dead_code)]
fn daughter_wavelet_multiply_scalar(
    input: &[Complex32],
    output: &mut [Complex32],
    mother: &[f32],
    scale: f32,
    imaginary: bool,
    double_sided: bool,
) {
    output.fill(Complex32::ZERO);

    let len = input.len();
    let endpoint = ((len as f32 / 2.0).min(len as f32 * 2.0 / scale)) as usize;
    let step = scale / 2.0;
    let maximum = len as f32 - 1.0;
    let imaginary_sign = if imaginary { -1.0 } else { 1.0 };

    for q in 0..endpoint {
        let tmp = maximum.min(step * q as f32) as usize;
        output[q] = Complex32::new(
            input[q].re * mother[tmp],
            input[q].im * mother[tmp] * imaginary_sign,
        );
    }

    if double_sided {
        let last = len - 1;
        for q in 0..endpoint {
            let tmp = maximum.min(step * q as f32) as usize;
            let dest = last - q;
            output[dest] = Complex32::new(
                input[dest].re * mother[tmp] * imaginary_sign,
                input[dest].im * mother[tmp],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use rustfft::num_complex::Complex32;

    use super::{Fcwt, daughter_wavelet_multiply, daughter_wavelet_multiply_scalar};
    use crate::{Morlet, ScaleType, Scales, Wavelet};

    #[derive(Clone, Debug)]
    struct DoubleSidedUnitWavelet {
        mother: Vec<f32>,
    }

    impl DoubleSidedUnitWavelet {
        fn new() -> Self {
            Self { mother: Vec::new() }
        }
    }

    impl Wavelet for DoubleSidedUnitWavelet {
        fn generate_frequency(&mut self, size: usize) {
            self.mother = vec![1.0; size];
        }

        fn generate_time(&mut self, _size: usize, _scale: f32) -> Vec<Complex32> {
            Vec::new()
        }

        fn support(&self, _scale: f32) -> usize {
            0
        }

        fn mother(&self) -> &[f32] {
            &self.mother
        }

        fn double_sided(&self) -> bool {
            true
        }
    }

    #[test]
    fn simd_multiply_matches_scalar_reference() {
        let input = (0..64)
            .map(|i| Complex32::new(i as f32 * 0.25, -(i as f32) * 0.125))
            .collect::<Vec<_>>();
        let mother = (0..64).map(|i| (i as f32 * 0.1).sin()).collect::<Vec<_>>();
        let mut simd = vec![Complex32::ZERO; 64];
        let mut scalar = vec![Complex32::ZERO; 64];

        daughter_wavelet_multiply(&input, &mut simd, &mother, 3.25, true, true);
        daughter_wavelet_multiply_scalar(&input, &mut scalar, &mother, 3.25, true, true);

        for (actual, expected) in simd.iter().zip(scalar.iter()) {
            assert_relative_eq!(actual.re, expected.re, epsilon = 1e-6);
            assert_relative_eq!(actual.im, expected.im, epsilon = 1e-6);
        }
    }

    #[test]
    fn transforms_impulse_signal() {
        let scales = Scales::new(ScaleType::LinearScales, 64, 4.0, 16.0, 4).unwrap();
        let mut fcwt = Fcwt::new(Morlet::new(2.0));
        let mut input = vec![0.0; 32];
        input[0] = 1.0;

        let output = fcwt.cwt_real(&input, &scales);

        assert_eq!(output.len(), input.len() * scales.len());
        assert!(
            output
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );
        assert!(output.iter().any(|value| value.norm() > 0.0));
    }

    #[test]
    fn transforms_sine_signal() {
        let scales = Scales::new(ScaleType::LinearFrequencies, 128, 4.0, 32.0, 8).unwrap();
        let mut fcwt = Fcwt::new(Morlet::new(2.0));
        let input = (0..64)
            .map(|i| (2.0 * std::f32::consts::PI * 8.0 * i as f32 / 128.0).sin())
            .collect::<Vec<_>>();

        let output = fcwt.cwt_real(&input, &scales);

        assert_eq!(output.len(), input.len() * scales.len());
        assert!(
            output
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );
    }

    #[test]
    fn default_output_is_inverse_fft_normalized() {
        let scales = Scales::new(ScaleType::LinearScales, 64, 4.0, 16.0, 3).unwrap();
        let input = (0..32)
            .map(|i| (2.0 * std::f32::consts::PI * 4.0 * i as f32 / 64.0).sin())
            .collect::<Vec<_>>();
        let mut default_fcwt = Fcwt::new(Morlet::new(2.0));
        let mut normalized_fcwt = Fcwt::new(Morlet::new(2.0)).with_normalization(true);

        let default_output = default_fcwt.cwt_real(&input, &scales);
        let normalized_output = normalized_fcwt.cwt_real(&input, &scales);

        for (actual, expected) in default_output.iter().zip(normalized_output.iter()) {
            assert_relative_eq!(actual.re, expected.re, epsilon = 1e-6);
            assert_relative_eq!(actual.im, expected.im, epsilon = 1e-6);
        }
    }

    #[test]
    fn normalization_can_be_disabled() {
        let scales = Scales::new(ScaleType::LinearScales, 64, 4.0, 16.0, 3).unwrap();
        let input = (0..32)
            .map(|i| (2.0 * std::f32::consts::PI * 4.0 * i as f32 / 64.0).sin())
            .collect::<Vec<_>>();
        let mut normalized_fcwt = Fcwt::new(Morlet::new(2.0));
        let mut raw_fcwt = Fcwt::new(Morlet::new(2.0)).with_normalization(false);

        let normalized_output = normalized_fcwt.cwt_real(&input, &scales);
        let raw_output = raw_fcwt.cwt_real(&input, &scales);
        let fft_size = input.len().next_power_of_two() as f32;

        for (normalized, raw) in normalized_output.iter().zip(raw_output.iter()) {
            assert_relative_eq!(raw.re, normalized.re * fft_size, epsilon = 1e-4);
            assert_relative_eq!(raw.im, normalized.im * fft_size, epsilon = 1e-4);
        }
    }

    #[test]
    fn complex_path_accepts_complex_samples() {
        let scales = Scales::new(ScaleType::LinearScales, 64, 4.0, 16.0, 3).unwrap();
        let mut fcwt = Fcwt::new(Morlet::new(2.0));
        let input = (0..16)
            .map(|i| Complex32::new(i as f32, -(i as f32)))
            .collect::<Vec<_>>();

        let output = fcwt.cwt_complex(&input, &scales);

        assert_eq!(output.len(), input.len() * scales.len());
        assert!(
            output
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );
    }

    #[test]
    fn complex_path_preserves_negative_frequency_content() {
        let scales = Scales::from_scales(128, vec![2.0]).unwrap();
        let mut fcwt = Fcwt::new(DoubleSidedUnitWavelet::new());
        let input = (0..128)
            .map(|i| {
                let phase = -2.0 * std::f32::consts::PI * 8.0 * i as f32 / 128.0;
                Complex32::new(phase.cos(), phase.sin())
            })
            .collect::<Vec<_>>();

        let output = fcwt.cwt_complex(&input, &scales);
        let max_norm = output
            .iter()
            .map(|value| value.norm())
            .fold(0.0_f32, f32::max);

        assert!(max_norm > 0.9);
    }
}
