use crate::TransformError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiscreteWavelet {
    Haar,
    Daubechies(usize),
    Symlet(usize),
}

impl DiscreteWavelet {
    pub fn name(self) -> String {
        match self {
            Self::Haar => "haar".to_string(),
            Self::Daubechies(n) => format!("db{n}"),
            Self::Symlet(n) => format!("sym{n}"),
        }
    }

    pub fn filter_bank(self) -> Result<WaveletFilterBank, TransformError> {
        match self {
            Self::Haar => WaveletFilterBank::orthogonal(self.name(), 1, &HAAR_LO),
            Self::Daubechies(2) => WaveletFilterBank::orthogonal(self.name(), 2, &DB2_LO),
            Self::Daubechies(4) => WaveletFilterBank::orthogonal(self.name(), 4, &DB4_LO),
            Self::Daubechies(6) => WaveletFilterBank::orthogonal(self.name(), 6, &DB6_LO),
            Self::Daubechies(8) => WaveletFilterBank::orthogonal(self.name(), 8, &DB8_LO),
            Self::Symlet(2) => WaveletFilterBank::orthogonal(self.name(), 2, &SYM2_LO),
            Self::Symlet(4) => WaveletFilterBank::orthogonal(self.name(), 4, &SYM4_LO),
            Self::Symlet(6) => WaveletFilterBank::orthogonal(self.name(), 6, &SYM6_LO),
            Self::Symlet(8) => WaveletFilterBank::orthogonal(self.name(), 8, &SYM8_LO),
            Self::Daubechies(_) | Self::Symlet(_) => Err(TransformError::InvalidWaveletFilterBank),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WaveletFilterBank {
    name: String,
    vanishing_moments: usize,
    support_len: usize,
    analysis_low: Vec<f32>,
    analysis_high: Vec<f32>,
    synthesis_low: Vec<f32>,
    synthesis_high: Vec<f32>,
}

impl WaveletFilterBank {
    pub fn new(
        name: impl Into<String>,
        vanishing_moments: usize,
        analysis_low: Vec<f32>,
        analysis_high: Vec<f32>,
        synthesis_low: Vec<f32>,
        synthesis_high: Vec<f32>,
    ) -> Result<Self, TransformError> {
        let support_len = analysis_low.len();
        if support_len == 0
            || analysis_high.len() != support_len
            || synthesis_low.len() != support_len
            || synthesis_high.len() != support_len
            || !analysis_low.iter().all(|value| value.is_finite())
            || !analysis_high.iter().all(|value| value.is_finite())
            || !synthesis_low.iter().all(|value| value.is_finite())
            || !synthesis_high.iter().all(|value| value.is_finite())
        {
            return Err(TransformError::InvalidWaveletFilterBank);
        }

        Ok(Self {
            name: name.into(),
            vanishing_moments,
            support_len,
            analysis_low,
            analysis_high,
            synthesis_low,
            synthesis_high,
        })
    }

    pub fn orthogonal(
        name: impl Into<String>,
        vanishing_moments: usize,
        analysis_low: &[f32],
    ) -> Result<Self, TransformError> {
        let analysis_high = qmf_highpass(analysis_low);
        Self::new(
            name,
            vanishing_moments,
            analysis_low.to_vec(),
            analysis_high.clone(),
            analysis_low.to_vec(),
            analysis_high,
        )
    }

    pub fn haar() -> Self {
        DiscreteWavelet::Haar
            .filter_bank()
            .expect("built-in Haar filter bank is valid")
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn vanishing_moments(&self) -> usize {
        self.vanishing_moments
    }

    pub fn support_len(&self) -> usize {
        self.support_len
    }

    pub fn analysis_low(&self) -> &[f32] {
        &self.analysis_low
    }

    pub fn analysis_high(&self) -> &[f32] {
        &self.analysis_high
    }

    pub fn synthesis_low(&self) -> &[f32] {
        &self.synthesis_low
    }

    pub fn synthesis_high(&self) -> &[f32] {
        &self.synthesis_high
    }
}

fn qmf_highpass(low: &[f32]) -> Vec<f32> {
    low.iter()
        .rev()
        .enumerate()
        .map(|(index, value)| if index % 2 == 0 { *value } else { -*value })
        .collect()
}

const HAAR_LO: [f32; 2] = [
    std::f32::consts::FRAC_1_SQRT_2,
    std::f32::consts::FRAC_1_SQRT_2,
];

const DB2_LO: [f32; 4] = [-0.12940952, 0.22414386, 0.8365163, 0.4829629];
const DB4_LO: [f32; 8] = [
    -0.010597402,
    0.03288301,
    0.030841382,
    -0.18703482,
    -0.02798377,
    0.6308808,
    0.71484655,
    0.23037781,
];
const DB6_LO: [f32; 12] = [
    -0.0010773011,
    0.0047772573,
    0.0005538422,
    -0.03158204,
    0.027522866,
    0.097501606,
    -0.12976687,
    -0.2262647,
    0.31525034,
    0.7511339,
    0.4946239,
    0.11154074,
];
const DB8_LO: [f32; 16] = [
    -0.00011747678,
    0.0006754494,
    -0.00039174038,
    -0.004870353,
    0.008746094,
    0.013981028,
    -0.044088256,
    -0.017369302,
    0.12874743,
    0.00047248456,
    -0.28401554,
    -0.015829105,
    0.5853547,
    0.67563075,
    0.3128716,
    0.05441584,
];

const SYM2_LO: [f32; 4] = [-0.12940952, 0.22414386, 0.8365163, 0.4829629];
const SYM4_LO: [f32; 8] = [
    -0.075765714,
    -0.029635528,
    0.49761868,
    0.8037388,
    0.2978578,
    -0.099219546,
    -0.012603967,
    0.0322231,
];
const SYM6_LO: [f32; 12] = [
    0.015404109,
    0.003490712,
    -0.117990114,
    -0.048311744,
    0.49105594,
    0.78764117,
    0.33792943,
    -0.07263752,
    -0.021060292,
    0.0447249,
    0.0017677118,
    -0.007800708,
];
const SYM8_LO: [f32; 16] = [
    -0.003382416,
    -0.0005421323,
    0.031695087,
    0.0076074875,
    -0.14329424,
    -0.06127336,
    0.48135966,
    0.77718574,
    0.3644419,
    -0.05194584,
    -0.02721903,
    0.04913718,
    0.003808752,
    -0.014952258,
    -0.0003029205,
    0.0018899504,
];

#[cfg(test)]
mod tests {
    use super::{DiscreteWavelet, WaveletFilterBank};
    use crate::TransformError;

    #[test]
    fn validates_custom_filter_banks() {
        assert_eq!(
            WaveletFilterBank::new("bad", 0, vec![1.0], vec![1.0, 2.0], vec![1.0], vec![1.0]),
            Err(TransformError::InvalidWaveletFilterBank)
        );
        assert_eq!(
            WaveletFilterBank::new("bad", 0, vec![f32::NAN], vec![1.0], vec![1.0], vec![1.0]),
            Err(TransformError::InvalidWaveletFilterBank)
        );
    }

    #[test]
    fn exposes_supported_wavelet_metadata() {
        let wavelet = DiscreteWavelet::Daubechies(4).filter_bank().unwrap();
        assert_eq!(wavelet.name(), "db4");
        assert_eq!(wavelet.vanishing_moments(), 4);
        assert_eq!(wavelet.support_len(), 8);
    }
}
