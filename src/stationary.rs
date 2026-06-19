use crate::discrete::{TransformError, circular_convolve_same, validate_power_of_two};
use crate::{DiscreteWavelet, WaveletFilterBank};

#[derive(Clone, Debug, PartialEq)]
pub struct SwtLevel {
    pub approximation: Vec<f32>,
    pub detail: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SwtCoefficients {
    input_len: usize,
    levels: Vec<SwtLevel>,
}

impl SwtCoefficients {
    pub fn input_len(&self) -> usize {
        self.input_len
    }

    pub fn levels(&self) -> &[SwtLevel] {
        &self.levels
    }

    pub fn into_levels(self) -> Vec<SwtLevel> {
        self.levels
    }

    pub fn from_levels(input_len: usize, levels: Vec<SwtLevel>) -> Self {
        Self { input_len, levels }
    }
}

#[derive(Clone, Debug)]
pub struct StationaryWaveletTransform {
    levels: usize,
    filter_bank: WaveletFilterBank,
}

impl StationaryWaveletTransform {
    pub fn new(levels: usize) -> Self {
        Self {
            levels,
            filter_bank: WaveletFilterBank::haar(),
        }
    }

    pub fn with_wavelet(levels: usize, wavelet: DiscreteWavelet) -> Result<Self, TransformError> {
        Ok(Self {
            levels,
            filter_bank: wavelet.filter_bank()?,
        })
    }

    pub fn with_filter_bank(levels: usize, filter_bank: WaveletFilterBank) -> Self {
        Self {
            levels,
            filter_bank,
        }
    }

    pub fn levels(&self) -> usize {
        self.levels
    }

    pub fn filter_bank(&self) -> &WaveletFilterBank {
        &self.filter_bank
    }

    pub fn decompose(&self, input: &[f32]) -> Result<SwtCoefficients, TransformError> {
        validate_power_of_two(input.len(), self.levels)?;

        let mut current = input.to_vec();
        let mut levels = Vec::with_capacity(self.levels);
        for level in 0..self.levels {
            let stride = 1 << level;
            let approximation =
                circular_convolve_same(&current, self.filter_bank.analysis_low(), stride);
            let detail = circular_convolve_same(&current, self.filter_bank.analysis_high(), stride);
            current = approximation.clone();
            levels.push(SwtLevel {
                approximation,
                detail,
            });
        }

        Ok(SwtCoefficients {
            input_len: input.len(),
            levels,
        })
    }

    pub fn reconstruct(&self, coefficients: &SwtCoefficients) -> Result<Vec<f32>, TransformError> {
        if coefficients.levels.len() != self.levels {
            return Err(TransformError::InvalidCoefficientTree);
        }
        validate_power_of_two(coefficients.input_len, self.levels)?;

        let mut current = coefficients
            .levels
            .last()
            .map(|level| level.approximation.clone())
            .unwrap_or_else(|| vec![0.0; coefficients.input_len]);

        if self.levels == 0 {
            return Ok(current);
        }

        for level_index in (0..self.levels).rev() {
            let level = &coefficients.levels[level_index];
            if level.approximation.len() != coefficients.input_len
                || level.detail.len() != coefficients.input_len
                || current.len() != coefficients.input_len
            {
                return Err(TransformError::InvalidCoefficientTree);
            }

            current = inverse_swt_level(
                &current,
                &level.detail,
                self.filter_bank.synthesis_low(),
                self.filter_bank.synthesis_high(),
                1 << level_index,
            );
        }

        Ok(current)
    }
}

fn inverse_swt_level(
    approximation: &[f32],
    detail: &[f32],
    low: &[f32],
    high: &[f32],
    stride: usize,
) -> Vec<f32> {
    let len = approximation.len();
    let mut output = vec![0.0; len];
    for (i, output_sample) in output.iter_mut().enumerate().take(len) {
        for tap in 0..low.len() {
            let sample = (i + len - (tap * stride) % len) % len;
            *output_sample += low[tap] * approximation[sample] + high[tap] * detail[sample];
        }
        *output_sample *= 0.5;
    }
    output
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::StationaryWaveletTransform;
    use crate::{DiscreteWavelet, TransformError};

    #[test]
    fn rejects_invalid_inputs() {
        assert_eq!(
            StationaryWaveletTransform::new(1).decompose(&[]),
            Err(TransformError::EmptyInput)
        );
        assert_eq!(
            StationaryWaveletTransform::new(1).decompose(&[1.0, 2.0, 3.0]),
            Err(TransformError::NonPowerOfTwo { len: 3 })
        );
    }

    #[test]
    fn preserves_lengths_at_each_level() {
        let coeffs = StationaryWaveletTransform::new(3)
            .decompose(&[1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0])
            .unwrap();

        assert_eq!(coeffs.levels().len(), 3);
        for level in coeffs.levels() {
            assert_eq!(level.approximation.len(), 8);
            assert_eq!(level.detail.len(), 8);
        }
    }

    #[test]
    fn first_level_matches_haar_with_periodic_wrap() {
        let coeffs = StationaryWaveletTransform::new(1)
            .decompose(&[1.0, 2.0, 3.0, 4.0])
            .unwrap();

        let level = &coeffs.levels()[0];
        assert_relative_eq!(
            level.approximation[0],
            3.0 * std::f32::consts::FRAC_1_SQRT_2
        );
        assert_relative_eq!(level.detail[0], -std::f32::consts::FRAC_1_SQRT_2);
        assert_relative_eq!(
            level.approximation[3],
            5.0 * std::f32::consts::FRAC_1_SQRT_2
        );
        assert_relative_eq!(level.detail[3], 3.0 * std::f32::consts::FRAC_1_SQRT_2);
    }

    #[test]
    fn reconstructs_input() {
        let input = [1.0, -2.0, 3.5, 4.25, -5.0, 6.0, 7.0, -8.0];
        let transform = StationaryWaveletTransform::new(3);
        let coeffs = transform.decompose(&input).unwrap();
        let reconstructed = transform.reconstruct(&coeffs).unwrap();

        for (actual, expected) in reconstructed.iter().zip(input) {
            assert_relative_eq!(*actual, expected, epsilon = 1e-5);
        }
    }

    #[test]
    fn reconstructs_input_with_supported_wavelets() {
        let input = [
            1.0, -2.0, 3.5, 4.25, -5.0, 6.0, 7.0, -8.0, 0.5, 2.25, -1.75, 9.0, 3.0, -4.0, 5.5, -6.5,
        ];
        let wavelets = [
            DiscreteWavelet::Daubechies(2),
            DiscreteWavelet::Daubechies(4),
            DiscreteWavelet::Daubechies(6),
            DiscreteWavelet::Daubechies(8),
            DiscreteWavelet::Symlet(2),
            DiscreteWavelet::Symlet(4),
            DiscreteWavelet::Symlet(6),
            DiscreteWavelet::Symlet(8),
        ];

        for wavelet in wavelets {
            let transform = StationaryWaveletTransform::with_wavelet(2, wavelet).unwrap();
            let coeffs = transform.decompose(&input).unwrap();
            let reconstructed = transform.reconstruct(&coeffs).unwrap();

            for (actual, expected) in reconstructed.iter().zip(input) {
                assert_relative_eq!(*actual, expected, epsilon = 5e-5);
            }
        }
    }

    #[test]
    fn circular_shift_shifts_coefficients() {
        let input = [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
        let shifted = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0];
        let transform = StationaryWaveletTransform::new(2);
        let coeffs = transform.decompose(&input).unwrap();
        let shifted_coeffs = transform.decompose(&shifted).unwrap();

        for (level, shifted_level) in coeffs.levels().iter().zip(shifted_coeffs.levels()) {
            for i in 0..input.len() {
                assert_relative_eq!(
                    level.approximation[i],
                    shifted_level.approximation[(i + 1) % input.len()],
                    epsilon = 1e-6
                );
                assert_relative_eq!(
                    level.detail[i],
                    shifted_level.detail[(i + 1) % input.len()],
                    epsilon = 1e-6
                );
            }
        }
    }
}
