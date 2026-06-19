use crate::{
    DiscreteWavelet, PacketBand, StationaryWaveletTransform, TransformError, WaveletPacketTransform,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransformKind {
    WaveletPacket,
    Stationary,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BasisSelectionCriterion {
    CoarsestDetailPdfShape,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BasisScore {
    pub wavelet: DiscreteWavelet,
    pub wavelet_name: String,
    pub transform_kind: TransformKind,
    pub criterion: BasisSelectionCriterion,
    pub score: f32,
    pub subband_level: usize,
    pub coefficient_count: usize,
    pub median_abs_deviation: f32,
    pub interquartile_range: f32,
    pub central_concentration: f32,
    pub sharpness: f32,
    pub pdf_peak: f32,
    pub pdf_center: f32,
    pub pdf_tail_mass: f32,
    pub pdf_entropy: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectedBasis {
    pub selected: DiscreteWavelet,
    pub selected_name: String,
    pub score: BasisScore,
    pub candidate_scores: Vec<BasisScore>,
}

pub fn score_basis(
    input: &[f32],
    levels: usize,
    wavelet: DiscreteWavelet,
    transform_kind: TransformKind,
) -> Result<BasisScore, TransformError> {
    let detail = coarsest_detail(input, levels, wavelet, transform_kind)?;
    score_detail(&detail, levels, wavelet, transform_kind)
}

pub fn select_basis(
    input: &[f32],
    levels: usize,
    candidates: &[DiscreteWavelet],
    transform_kind: TransformKind,
    criterion: BasisSelectionCriterion,
) -> Result<SelectedBasis, TransformError> {
    if candidates.is_empty() {
        return Err(TransformError::InvalidWaveletFilterBank);
    }

    let mut candidate_scores = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let score = match criterion {
            BasisSelectionCriterion::CoarsestDetailPdfShape => {
                score_basis(input, levels, *candidate, transform_kind)?
            }
        };
        candidate_scores.push(score);
    }

    let score = candidate_scores
        .iter()
        .max_by(|left, right| left.score.total_cmp(&right.score))
        .expect("candidate_scores is non-empty")
        .clone();

    Ok(SelectedBasis {
        selected: score.wavelet,
        selected_name: score.wavelet_name.clone(),
        score,
        candidate_scores,
    })
}

fn coarsest_detail(
    input: &[f32],
    levels: usize,
    wavelet: DiscreteWavelet,
    transform_kind: TransformKind,
) -> Result<Vec<f32>, TransformError> {
    match transform_kind {
        TransformKind::WaveletPacket => {
            let tree = WaveletPacketTransform::with_wavelet(levels, wavelet)?.decompose(input)?;
            let mut path = Vec::with_capacity(levels);
            path.extend(std::iter::repeat_n(
                PacketBand::Approximation,
                levels.saturating_sub(1),
            ));
            path.push(PacketBand::Detail);
            tree.leaves()
                .iter()
                .find(|node| node.path == path)
                .map(|node| node.coefficients.clone())
                .ok_or(TransformError::InvalidCoefficientTree)
        }
        TransformKind::Stationary => {
            let coefficients =
                StationaryWaveletTransform::with_wavelet(levels, wavelet)?.decompose(input)?;
            coefficients
                .levels()
                .last()
                .map(|level| level.detail.clone())
                .ok_or(TransformError::LevelTooDeep {
                    levels,
                    max_levels: 0,
                })
        }
    }
}

fn score_detail(
    detail: &[f32],
    levels: usize,
    wavelet: DiscreteWavelet,
    transform_kind: TransformKind,
) -> Result<BasisScore, TransformError> {
    if detail.len() < 4 || !detail.iter().all(|value| value.is_finite()) {
        return Err(TransformError::InvalidCoefficientTree);
    }

    let abs_values = detail.iter().map(|value| value.abs()).collect::<Vec<_>>();
    let median_abs_deviation = median(abs_values.clone());
    let q1 = percentile(abs_values.clone(), 0.25);
    let q3 = percentile(abs_values.clone(), 0.75);
    let interquartile_range = q3 - q1;
    let width = median_abs_deviation + interquartile_range + f32::EPSILON;
    let central_threshold = q1.max(f32::EPSILON);
    let central_concentration = abs_values
        .iter()
        .filter(|value| **value <= central_threshold)
        .count() as f32
        / abs_values.len() as f32;

    let mean_square = detail.iter().map(|value| value * value).sum::<f32>() / detail.len() as f32;
    let mean_fourth = detail.iter().map(|value| value.powi(4)).sum::<f32>() / detail.len() as f32;
    let sharpness = if mean_square > f32::EPSILON {
        mean_fourth / (mean_square * mean_square)
    } else {
        0.0
    };
    let pdf_shape = estimate_pdf_shape(detail);
    let peak_to_tail = pdf_shape.pdf_peak / (pdf_shape.pdf_tail_mass + f32::EPSILON);
    let score = central_concentration / width
        + sharpness.ln_1p()
        + peak_to_tail.ln_1p()
        + (1.0 - pdf_shape.pdf_entropy);

    Ok(BasisScore {
        wavelet,
        wavelet_name: wavelet.name(),
        transform_kind,
        criterion: BasisSelectionCriterion::CoarsestDetailPdfShape,
        score,
        subband_level: levels,
        coefficient_count: detail.len(),
        median_abs_deviation,
        interquartile_range,
        central_concentration,
        sharpness,
        pdf_peak: pdf_shape.pdf_peak,
        pdf_center: pdf_shape.pdf_center,
        pdf_tail_mass: pdf_shape.pdf_tail_mass,
        pdf_entropy: pdf_shape.pdf_entropy,
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PdfShape {
    pdf_peak: f32,
    pdf_center: f32,
    pdf_tail_mass: f32,
    pdf_entropy: f32,
}

fn estimate_pdf_shape(values: &[f32]) -> PdfShape {
    let max_abs = values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max);
    if max_abs <= f32::EPSILON {
        return PdfShape {
            pdf_peak: 1.0,
            pdf_center: 1.0,
            pdf_tail_mass: 0.0,
            pdf_entropy: 0.0,
        };
    }

    let bin_count = ((values.len() as f32).sqrt().round() as usize).clamp(8, 64);
    let min = -max_abs;
    let width = (2.0 * max_abs) / bin_count as f32;
    let mut bins = vec![0usize; bin_count];

    for value in values {
        let scaled = ((*value - min) / width).floor();
        let index = (scaled as isize).clamp(0, bin_count as isize - 1) as usize;
        bins[index] += 1;
    }

    let count = values.len() as f32;
    let probabilities = bins
        .iter()
        .map(|bin| *bin as f32 / count)
        .collect::<Vec<_>>();
    let pdf_peak = probabilities.iter().copied().fold(0.0_f32, f32::max);
    let center_index = ((0.0 - min) / width).floor() as usize;
    let pdf_center = probabilities[center_index.min(bin_count - 1)];
    let pdf_entropy = normalized_entropy(&probabilities);

    let tail_start = (0.75 * bin_count as f32).floor() as usize;
    let pdf_tail_mass = probabilities
        .iter()
        .enumerate()
        .filter(|(index, _)| *index >= tail_start || *index < bin_count - tail_start)
        .map(|(_, probability)| *probability)
        .sum();

    PdfShape {
        pdf_peak,
        pdf_center,
        pdf_tail_mass,
        pdf_entropy,
    }
}

fn normalized_entropy(probabilities: &[f32]) -> f32 {
    let active_bins = probabilities
        .iter()
        .filter(|probability| **probability > 0.0)
        .count();
    if active_bins <= 1 {
        return 0.0;
    }

    let entropy = probabilities
        .iter()
        .filter(|probability| **probability > 0.0)
        .map(|probability| -probability * probability.ln())
        .sum::<f32>();
    entropy / (active_bins as f32).ln()
}

fn median(mut values: Vec<f32>) -> f32 {
    percentile_sorted(&mut values, 0.5)
}

fn percentile(mut values: Vec<f32>, p: f32) -> f32 {
    percentile_sorted(&mut values, p)
}

fn percentile_sorted(values: &mut [f32], p: f32) -> f32 {
    values.sort_by(|left, right| left.total_cmp(right));
    let index = ((values.len() - 1) as f32 * p).round() as usize;
    values[index]
}

#[cfg(test)]
mod tests {
    use super::{
        BasisSelectionCriterion, TransformKind, estimate_pdf_shape, score_basis, select_basis,
    };
    use crate::{DiscreteWavelet, TransformError};
    use approx::assert_relative_eq;

    #[test]
    fn scores_return_shape_diagnostics() {
        let input = [
            1.0, 1.0, 1.0, 8.0, 1.0, 1.0, 1.0, -8.0, 1.0, 1.0, 1.0, 8.0, 1.0, 1.0, 1.0, -8.0,
        ];
        let score = score_basis(
            &input,
            2,
            DiscreteWavelet::Haar,
            TransformKind::WaveletPacket,
        )
        .unwrap();

        assert_eq!(score.wavelet_name, "haar");
        assert_eq!(score.subband_level, 2);
        assert_eq!(score.coefficient_count, 4);
        assert!(score.score.is_finite());
        assert!(score.pdf_peak.is_finite());
        assert!(score.pdf_center.is_finite());
        assert!(score.pdf_tail_mass.is_finite());
        assert!(score.pdf_entropy.is_finite());
    }

    #[test]
    fn selector_returns_candidate_scores() {
        let input = [
            0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, -4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, -4.0,
        ];
        let selected = select_basis(
            &input,
            2,
            &[DiscreteWavelet::Haar, DiscreteWavelet::Daubechies(2)],
            TransformKind::Stationary,
            BasisSelectionCriterion::CoarsestDetailPdfShape,
        )
        .unwrap();

        assert_eq!(selected.candidate_scores.len(), 2);
        assert!(selected.score.score.is_finite());
    }

    #[test]
    fn score_rejects_too_short_coarsest_detail_bands() {
        assert_eq!(
            score_basis(
                &[1.0, 2.0, 3.0, 4.0],
                2,
                DiscreteWavelet::Haar,
                TransformKind::WaveletPacket
            ),
            Err(TransformError::InvalidCoefficientTree)
        );
    }

    #[test]
    fn pdf_shape_prefers_peaked_sparse_distributions() {
        let sparse = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, -4.0, 5.0, -5.0,
        ];
        let broad = [
            -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
        ];

        let sparse_shape = estimate_pdf_shape(&sparse);
        let broad_shape = estimate_pdf_shape(&broad);

        assert!(sparse_shape.pdf_peak > broad_shape.pdf_peak);
        assert!(sparse_shape.pdf_center > broad_shape.pdf_center);
        assert!(sparse_shape.pdf_entropy < broad_shape.pdf_entropy);
    }

    #[test]
    fn selector_can_choose_basis_with_more_peaked_coarsest_detail() {
        let input = [
            0.0, 0.25, 0.5, 0.75, 8.0, 0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -0.75, -8.0, -0.75, -0.5,
            -0.25, 0.0, 0.25, 0.5, 0.75, 8.0, 0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -0.75, -8.0,
            -0.75, -0.5, -0.25,
        ];
        let selected = select_basis(
            &input,
            3,
            &[
                DiscreteWavelet::Haar,
                DiscreteWavelet::Daubechies(2),
                DiscreteWavelet::Symlet(4),
            ],
            TransformKind::WaveletPacket,
            BasisSelectionCriterion::CoarsestDetailPdfShape,
        )
        .unwrap();

        let selected_score = selected.score.score;
        for score in selected.candidate_scores {
            assert!(selected_score >= score.score);
        }
    }

    #[test]
    fn all_zero_detail_has_degenerate_pdf_at_center() {
        let shape = estimate_pdf_shape(&[0.0; 16]);
        assert_relative_eq!(shape.pdf_peak, 1.0);
        assert_relative_eq!(shape.pdf_center, 1.0);
        assert_relative_eq!(shape.pdf_tail_mass, 0.0);
        assert_relative_eq!(shape.pdf_entropy, 0.0);
    }
}
