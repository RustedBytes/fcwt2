use rustfft::num_complex::Complex32;

use crate::discrete::{TransformError, reflect_index, validate_power_of_two};

const NEAR_SYM_A_H0O: [f32; 5] = [-0.05, 0.25, 0.6, 0.25, -0.05];
const NEAR_SYM_A_G0O: [f32; 7] = [
    -0.010_714_286,
    -0.053_571_43,
    0.260_714_3,
    0.607_142_87,
    0.260_714_3,
    -0.053_571_43,
    -0.010_714_286,
];
const NEAR_SYM_A_H1O: [f32; 7] = [
    0.010_714_286,
    -0.053_571_43,
    -0.260_714_3,
    0.607_142_87,
    -0.260_714_3,
    -0.053_571_43,
    0.010_714_286,
];
const NEAR_SYM_A_G1O: [f32; 5] = [-0.05, -0.25, 0.6, -0.25, -0.05];

const QSHIFT_A_H0A: [f32; 10] = [
    0.051_130_407,
    -0.013_975_37,
    -0.109_836_05,
    0.263_839_57,
    0.766_628_44,
    0.563_655_73,
    0.000_873_622_7,
    -0.100_231_22,
    -0.001_689_681_3,
    -0.006_181_882,
];
const QSHIFT_A_H0B: [f32; 10] = [
    -0.006_181_882,
    -0.001_689_681_3,
    -0.100_231_22,
    0.000_873_622_7,
    0.563_655_73,
    0.766_628_44,
    0.263_839_57,
    -0.109_836_05,
    -0.013_975_37,
    0.051_130_407,
];
const QSHIFT_A_G0A: [f32; 10] = QSHIFT_A_H0B;
const QSHIFT_A_G0B: [f32; 10] = QSHIFT_A_H0A;
const QSHIFT_A_H1A: [f32; 10] = [
    -0.006_181_882,
    0.001_689_681_3,
    -0.100_231_22,
    -0.000_873_622_7,
    0.563_655_73,
    -0.766_628_44,
    0.263_839_57,
    0.109_836_05,
    -0.013_975_37,
    -0.051_130_407,
];
const QSHIFT_A_H1B: [f32; 10] = [
    -0.051_130_407,
    -0.013_975_37,
    0.109_836_05,
    0.263_839_57,
    -0.766_628_44,
    0.563_655_73,
    -0.000_873_622_7,
    -0.100_231_22,
    0.001_689_681_3,
    -0.006_181_882,
];
const QSHIFT_A_G1A: [f32; 10] = QSHIFT_A_H1B;
const QSHIFT_A_G1B: [f32; 10] = QSHIFT_A_H1A;

#[derive(Clone, Debug, PartialEq)]
pub struct DtcwtLevel {
    pub detail: Vec<Complex32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DtcwtTree {
    lowpass: Vec<f32>,
    highpasses: Vec<DtcwtLevel>,
}

impl DtcwtTree {
    pub fn lowpass(&self) -> &[f32] {
        &self.lowpass
    }

    pub fn highpasses(&self) -> &[DtcwtLevel] {
        &self.highpasses
    }
}

#[derive(Clone, Debug)]
pub struct DualTreeComplexWaveletTransform {
    levels: usize,
}

impl DualTreeComplexWaveletTransform {
    pub fn new(levels: usize) -> Self {
        Self { levels }
    }

    pub fn levels(&self) -> usize {
        self.levels
    }

    pub fn decompose(&self, input: &[f32]) -> Result<DtcwtTree, TransformError> {
        validate_power_of_two(input.len(), self.levels)?;

        if self.levels == 0 {
            return Ok(DtcwtTree {
                lowpass: input.to_vec(),
                highpasses: Vec::new(),
            });
        }

        let hi = colfilter(input, &NEAR_SYM_A_H1O);
        let mut lo = colfilter(input, &NEAR_SYM_A_H0O);
        let mut highpasses = vec![DtcwtLevel {
            detail: real_to_complex_quads(&hi),
        }];

        for _ in 1..self.levels {
            if lo.len() % 4 != 0 {
                lo = extend_once(&lo);
            }
            let hi = coldfilt(&lo, &QSHIFT_A_H1B, &QSHIFT_A_H1A)?;
            lo = coldfilt(&lo, &QSHIFT_A_H0B, &QSHIFT_A_H0A)?;
            highpasses.push(DtcwtLevel {
                detail: real_to_complex_quads(&hi),
            });
        }

        Ok(DtcwtTree {
            lowpass: lo,
            highpasses,
        })
    }

    pub fn reconstruct(&self, tree: &DtcwtTree) -> Result<Vec<f32>, TransformError> {
        if tree.highpasses.len() != self.levels {
            return Err(TransformError::InvalidCoefficientTree);
        }

        if self.levels == 0 {
            validate_power_of_two(tree.lowpass.len(), 0)?;
            return Ok(tree.lowpass.clone());
        }

        let mut lo = tree.lowpass.clone();

        for level in (1..self.levels).rev() {
            let hi = complex_quads_to_real(&tree.highpasses[level].detail);
            if hi.len() != lo.len() {
                return Err(TransformError::InvalidCoefficientTree);
            }

            lo = add_vectors(
                &colifilt(&lo, &QSHIFT_A_G0B, &QSHIFT_A_G0A)?,
                &colifilt(&hi, &QSHIFT_A_G1B, &QSHIFT_A_G1A)?,
            )?;

            let previous_high_len = tree.highpasses[level - 1].detail.len() * 2;
            if lo.len() != previous_high_len {
                if lo.len() < 2 {
                    return Err(TransformError::InvalidCoefficientTree);
                }
                lo = lo[1..lo.len() - 1].to_vec();
            }
            if lo.len() != previous_high_len {
                return Err(TransformError::InvalidCoefficientTree);
            }
        }

        let hi = complex_quads_to_real(&tree.highpasses[0].detail);
        if hi.len() != lo.len() {
            return Err(TransformError::InvalidCoefficientTree);
        }
        add_vectors(
            &colfilter(&lo, &NEAR_SYM_A_G0O),
            &colfilter(&hi, &NEAR_SYM_A_G1O),
        )
    }
}

fn extend_once(input: &[f32]) -> Vec<f32> {
    let mut output = Vec::with_capacity(input.len() + 2);
    output.push(input[0]);
    output.extend_from_slice(input);
    output.push(input[input.len() - 1]);
    output
}

fn add_vectors(left: &[f32], right: &[f32]) -> Result<Vec<f32>, TransformError> {
    if left.len() != right.len() {
        return Err(TransformError::InvalidCoefficientTree);
    }

    Ok(left.iter().zip(right).map(|(a, b)| a + b).collect())
}

fn real_to_complex_quads(input: &[f32]) -> Vec<Complex32> {
    input
        .chunks_exact(2)
        .map(|chunk| Complex32::new(chunk[0], chunk[1]))
        .collect()
}

fn complex_quads_to_real(input: &[Complex32]) -> Vec<f32> {
    let mut output = Vec::with_capacity(input.len() * 2);
    for value in input {
        output.push(value.re);
        output.push(value.im);
    }
    output
}

fn colfilter(input: &[f32], filter: &[f32]) -> Vec<f32> {
    let len = input.len();
    let filter_len = filter.len();
    let half = filter_len / 2;
    let extension = (-(half as isize)..len as isize + half as isize)
        .map(|index| input[reflect_index(index, -0.5, len as f32 - 0.5)])
        .collect::<Vec<_>>();

    column_convolve_centered(&extension, filter)
}

fn coldfilt(input: &[f32], ha: &[f32], hb: &[f32]) -> Result<Vec<f32>, TransformError> {
    if input.len() % 4 != 0 || ha.len() != hb.len() || ha.len() % 2 != 0 {
        return Err(TransformError::InvalidCoefficientTree);
    }

    let len = input.len();
    let filter_len = ha.len();
    let extension = (-(filter_len as isize)..len as isize + filter_len as isize)
        .map(|index| input[reflect_index(index, -0.5, len as f32 - 0.5)])
        .collect::<Vec<_>>();

    let hao = even_taps(ha);
    let hae = odd_taps(ha);
    let hbo = even_taps(hb);
    let hbe = odd_taps(hb);
    let dot_positive = dot(ha, hb) > 0.0;
    let mut output = vec![0.0; len / 2];

    let t = (5..len + 2 * filter_len - 2).step_by(4).collect::<Vec<_>>();
    let ya = add_plain(
        &column_convolve_centered(&select_by_offsets(&extension, &t, -1), &hao),
        &column_convolve_centered(&select_by_offsets(&extension, &t, -3), &hae),
    );
    let yb = add_plain(
        &column_convolve_centered(&select_by_offsets(&extension, &t, 0), &hbo),
        &column_convolve_centered(&select_by_offsets(&extension, &t, -2), &hbe),
    );

    for pair in 0..ya.len() {
        if dot_positive {
            output[2 * pair] = ya[pair];
            output[2 * pair + 1] = yb[pair];
        } else {
            output[2 * pair + 1] = ya[pair];
            output[2 * pair] = yb[pair];
        }
    }

    Ok(output)
}

fn colifilt(input: &[f32], ha: &[f32], hb: &[f32]) -> Result<Vec<f32>, TransformError> {
    if input.len() % 2 != 0 || ha.len() != hb.len() || ha.len() % 2 != 0 {
        return Err(TransformError::InvalidCoefficientTree);
    }

    let len = input.len();
    let filter_len = ha.len();
    let half = filter_len / 2;
    let extension = (-(half as isize)..len as isize + half as isize)
        .map(|index| input[reflect_index(index, -0.5, len as f32 - 0.5)])
        .collect::<Vec<_>>();

    let hao = even_taps(ha);
    let hae = odd_taps(ha);
    let hbo = even_taps(hb);
    let hbe = odd_taps(hb);
    let dot_positive = dot(ha, hb) > 0.0;
    let mut output = vec![0.0; len * 2];

    if half % 2 == 0 {
        let t = (3..len + filter_len).step_by(2).collect::<Vec<_>>();
        let (ta, tb) = if dot_positive {
            (t.clone(), offset_values(&t, -1))
        } else {
            (offset_values(&t, -1), t)
        };
        let y0 = column_convolve_centered(&select_by_offsets(&extension, &tb, -2), &hae);
        let y1 = column_convolve_centered(&select_by_offsets(&extension, &ta, -2), &hbe);
        let y2 = column_convolve_centered(&select_by_offsets(&extension, &tb, 0), &hao);
        let y3 = column_convolve_centered(&select_by_offsets(&extension, &ta, 0), &hbo);
        for idx in 0..y0.len() {
            let s = idx * 4;
            output[s] = y0[idx];
            output[s + 1] = y1[idx];
            output[s + 2] = y2[idx];
            output[s + 3] = y3[idx];
        }
    } else {
        let t = (2..len + filter_len - 1).step_by(2).collect::<Vec<_>>();
        let (ta, tb) = if dot_positive {
            (t.clone(), offset_values(&t, -1))
        } else {
            (offset_values(&t, -1), t)
        };
        let y0 = column_convolve_centered(&select_by_offsets(&extension, &tb, 0), &hao);
        let y1 = column_convolve_centered(&select_by_offsets(&extension, &ta, 0), &hbo);
        let y2 = column_convolve_centered(&select_by_offsets(&extension, &tb, 0), &hae);
        let y3 = column_convolve_centered(&select_by_offsets(&extension, &ta, 0), &hbe);
        for idx in 0..y0.len() {
            let s = idx * 4;
            output[s] = y0[idx];
            output[s + 1] = y1[idx];
            output[s + 2] = y2[idx];
            output[s + 3] = y3[idx];
        }
    }

    Ok(output)
}

fn column_convolve_centered(input: &[f32], filter: &[f32]) -> Vec<f32> {
    let output_len = input.len().abs_diff(filter.len()) + 1;
    let full_len = input.len() + filter.len() - 1;
    let start = (full_len - output_len) / 2;

    (0..output_len)
        .map(|out| {
            let full_index = start + out;
            filter
                .iter()
                .enumerate()
                .filter_map(|(tap, coeff)| {
                    full_index
                        .checked_sub(tap)
                        .filter(|sample| *sample < input.len())
                        .map(|sample| coeff * input[sample])
                })
                .sum()
        })
        .collect()
}

fn select_by_offsets(input: &[f32], indices: &[usize], offset: isize) -> Vec<f32> {
    indices
        .iter()
        .map(|index| input[(*index as isize + offset) as usize])
        .collect()
}

fn offset_values(indices: &[usize], offset: isize) -> Vec<usize> {
    indices
        .iter()
        .map(|index| (*index as isize + offset) as usize)
        .collect()
}

fn add_plain(left: &[f32], right: &[f32]) -> Vec<f32> {
    left.iter().zip(right).map(|(a, b)| a + b).collect()
}

fn even_taps(filter: &[f32]) -> Vec<f32> {
    filter.iter().step_by(2).copied().collect()
}

fn odd_taps(filter: &[f32]) -> Vec<f32> {
    filter.iter().skip(1).step_by(2).copied().collect()
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::{DualTreeComplexWaveletTransform, colfilter};
    use crate::TransformError;

    #[test]
    fn rejects_invalid_inputs() {
        assert_eq!(
            DualTreeComplexWaveletTransform::new(1).decompose(&[]),
            Err(TransformError::EmptyInput)
        );
        assert_eq!(
            DualTreeComplexWaveletTransform::new(1).decompose(&[1.0, 2.0, 3.0]),
            Err(TransformError::NonPowerOfTwo { len: 3 })
        );
    }

    #[test]
    fn colfilter_preserves_length_for_odd_filters() {
        let output = colfilter(&[1.0, 2.0, 3.0, 4.0], &[0.25, 0.5, 0.25]);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn decomposes_to_expected_shapes() {
        let input = (0..32).map(|i| (i as f32 * 0.25).sin()).collect::<Vec<_>>();
        let tree = DualTreeComplexWaveletTransform::new(3)
            .decompose(&input)
            .unwrap();

        assert_eq!(tree.highpasses().len(), 3);
        assert_eq!(tree.highpasses()[0].detail.len(), 16);
        assert_eq!(tree.highpasses()[1].detail.len(), 8);
        assert_eq!(tree.highpasses()[2].detail.len(), 4);
        assert_eq!(tree.lowpass().len(), 8);
        assert!(
            tree.highpasses()[0]
                .detail
                .iter()
                .any(|value| value.im != 0.0)
        );
    }

    #[test]
    fn reconstructs_input() {
        let input = (0..64)
            .map(|i| ((i as f32) * 0.17).sin() + 0.25 * ((i as f32) * 0.41).cos())
            .collect::<Vec<_>>();
        let transform = DualTreeComplexWaveletTransform::new(4);
        let tree = transform.decompose(&input).unwrap();
        let reconstructed = transform.reconstruct(&tree).unwrap();

        assert_eq!(reconstructed.len(), input.len());
        for (actual, expected) in reconstructed.iter().zip(input) {
            assert_relative_eq!(*actual, expected, epsilon = 2e-4);
        }
    }
}
