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
    let score = central_concentration / width + sharpness.ln_1p();

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
    })
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
    use super::{BasisSelectionCriterion, TransformKind, score_basis, select_basis};
    use crate::{DiscreteWavelet, TransformError};

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
}
