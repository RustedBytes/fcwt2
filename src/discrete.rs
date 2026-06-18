#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransformError {
    EmptyInput,
    NonPowerOfTwo { len: usize },
    LevelTooDeep { levels: usize, max_levels: usize },
    InvalidCoefficientTree,
}

pub(crate) const FRAC_1_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;
pub(crate) const HAAR_LO: [f32; 2] = [FRAC_1_SQRT_2, FRAC_1_SQRT_2];
pub(crate) const HAAR_HI: [f32; 2] = [FRAC_1_SQRT_2, -FRAC_1_SQRT_2];

pub(crate) fn validate_power_of_two(
    input_len: usize,
    levels: usize,
) -> Result<usize, TransformError> {
    if input_len == 0 {
        return Err(TransformError::EmptyInput);
    }
    if !input_len.is_power_of_two() {
        return Err(TransformError::NonPowerOfTwo { len: input_len });
    }

    let max_levels = input_len.trailing_zeros() as usize;
    if levels > max_levels {
        return Err(TransformError::LevelTooDeep { levels, max_levels });
    }

    Ok(max_levels)
}

pub(crate) fn circular_convolve_same(input: &[f32], filter: &[f32], stride: usize) -> Vec<f32> {
    let len = input.len();
    (0..len)
        .map(|sample| {
            filter
                .iter()
                .enumerate()
                .map(|(tap, coeff)| coeff * input[(sample + tap * stride) % len])
                .sum()
        })
        .collect()
}

pub(crate) fn circular_downsample(
    input: &[f32],
    low: &[f32],
    high: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let out_len = input.len() / 2;
    let mut approx = vec![0.0; out_len];
    let mut detail = vec![0.0; out_len];

    for out in 0..out_len {
        for tap in 0..low.len() {
            let sample = input[(2 * out + tap) % input.len()];
            approx[out] += low[tap] * sample;
            detail[out] += high[tap] * sample;
        }
    }

    (approx, detail)
}

pub(crate) fn circular_upsample(
    approx: &[f32],
    detail: &[f32],
    low: &[f32],
    high: &[f32],
) -> Result<Vec<f32>, TransformError> {
    if approx.len() != detail.len() {
        return Err(TransformError::InvalidCoefficientTree);
    }

    let len = approx.len() * 2;
    let mut output = vec![0.0; len];
    for i in 0..approx.len() {
        for tap in 0..low.len() {
            let sample = (2 * i + tap) % len;
            output[sample] += low[tap] * approx[i] + high[tap] * detail[i];
        }
    }

    Ok(output)
}

pub(crate) fn reflect_index(index: isize, low: f32, high: f32) -> usize {
    let range = high - low;
    if range == 0.0 {
        return low.round() as usize;
    }

    let mut value = index as f32;
    loop {
        if value < low {
            value = 2.0 * low - value;
        } else if value > high {
            value = 2.0 * high - value;
        } else {
            return value.round() as usize;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        HAAR_HI, HAAR_LO, TransformError, circular_downsample, circular_upsample,
        validate_power_of_two,
    };
    use approx::assert_relative_eq;

    #[test]
    fn validates_lengths_and_levels() {
        assert_eq!(validate_power_of_two(0, 0), Err(TransformError::EmptyInput));
        assert_eq!(
            validate_power_of_two(6, 1),
            Err(TransformError::NonPowerOfTwo { len: 6 })
        );
        assert_eq!(
            validate_power_of_two(8, 4),
            Err(TransformError::LevelTooDeep {
                levels: 4,
                max_levels: 3
            })
        );
        assert_eq!(validate_power_of_two(8, 3), Ok(3));
    }

    #[test]
    fn haar_downsample_upsample_reconstructs() {
        let input = [1.0, 2.0, -3.0, 4.0, 5.0, -6.0, 7.0, 8.0];
        let (approx, detail) = circular_downsample(&input, &HAAR_LO, &HAAR_HI);
        let reconstructed = circular_upsample(&approx, &detail, &HAAR_LO, &HAAR_HI).unwrap();

        for (actual, expected) in reconstructed.iter().zip(input) {
            assert_relative_eq!(*actual, expected, epsilon = 1e-6);
        }
    }
}
