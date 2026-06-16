#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScaleType {
    LinearScales,
    LogScales,
    LinearFrequencies,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ScaleError {
    FrequencyAboveNyquist { frequency: f32, sample_rate: usize },
    EmptyScaleSet,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Scales {
    sample_rate: usize,
    scales: Vec<f32>,
}

impl Scales {
    pub fn new(
        scale_type: ScaleType,
        sample_rate: usize,
        f0: f32,
        f1: f32,
        nscales: usize,
    ) -> Result<Self, ScaleError> {
        if nscales == 0 {
            return Err(ScaleError::EmptyScaleSet);
        }
        check_nyquist(f1, sample_rate)?;

        let scales = match scale_type {
            ScaleType::LinearScales => linear_scale_array(sample_rate, f0, f1, nscales),
            ScaleType::LogScales => log_scale_array(2.0, sample_rate, f0, f1, nscales),
            ScaleType::LinearFrequencies => linear_frequency_array(sample_rate, f0, f1, nscales),
        };

        Ok(Self {
            sample_rate,
            scales,
        })
    }

    pub fn from_scales(sample_rate: usize, scales: Vec<f32>) -> Result<Self, ScaleError> {
        if scales.is_empty() {
            return Err(ScaleError::EmptyScaleSet);
        }

        Ok(Self {
            sample_rate,
            scales,
        })
    }

    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    pub fn len(&self) -> usize {
        self.scales.len()
    }

    pub fn is_empty(&self) -> bool {
        self.scales.is_empty()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.scales
    }

    pub fn frequencies(&self) -> Vec<f32> {
        self.scales
            .iter()
            .map(|scale| self.sample_rate as f32 / scale)
            .collect()
    }
}

fn check_nyquist(frequency: f32, sample_rate: usize) -> Result<(), ScaleError> {
    if frequency > sample_rate as f32 / 2.0 {
        Err(ScaleError::FrequencyAboveNyquist {
            frequency,
            sample_rate,
        })
    } else {
        Ok(())
    }
}

fn log_scale_array(base: f32, sample_rate: usize, f0: f32, f1: f32, nscales: usize) -> Vec<f32> {
    let s0 = sample_rate as f32 / f1;
    let s1 = sample_rate as f32 / f0;
    let power0 = s0.log(base);
    let power1 = s1.log(base);
    let dpower = power1 - power0;

    if nscales == 1 {
        return vec![base.powf(power0)];
    }

    (0..nscales)
        .map(|i| {
            let log_power = power0 + (dpower / (nscales - 1) as f32) * i as f32;
            base.powf(log_power)
        })
        .collect()
}

fn linear_frequency_array(sample_rate: usize, f0: f32, f1: f32, nscales: usize) -> Vec<f32> {
    let df = f1 - f0;
    let mut scales = vec![0.0; nscales];

    for i in 0..nscales {
        scales[nscales - i - 1] = sample_rate as f32 / (f0 + df / nscales as f32 * i as f32);
    }

    scales
}

fn linear_scale_array(sample_rate: usize, f0: f32, f1: f32, nscales: usize) -> Vec<f32> {
    let s0 = sample_rate as f32 / f1;
    let s1 = sample_rate as f32 / f0;
    let ds = s1 - s0;

    (0..nscales)
        .map(|i| s0 + ds / nscales as f32 * i as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::{ScaleError, ScaleType, Scales};

    #[test]
    fn creates_linear_scales() {
        let scales = Scales::new(ScaleType::LinearScales, 100, 10.0, 50.0, 4).unwrap();
        assert_relative_eq!(scales.as_slice()[0], 2.0);
        assert_relative_eq!(scales.as_slice()[1], 4.0);
        assert_relative_eq!(scales.as_slice()[2], 6.0);
        assert_relative_eq!(scales.as_slice()[3], 8.0);
    }

    #[test]
    fn creates_log_scales() {
        let scales = Scales::new(ScaleType::LogScales, 100, 10.0, 50.0, 3).unwrap();
        assert_relative_eq!(scales.as_slice()[0], 2.0, epsilon = 1e-6);
        assert_relative_eq!(scales.as_slice()[1], (20.0_f32).sqrt(), epsilon = 1e-6);
        assert_relative_eq!(scales.as_slice()[2], 10.0, epsilon = 1e-5);
    }

    #[test]
    fn creates_linear_frequencies() {
        let scales = Scales::new(ScaleType::LinearFrequencies, 100, 10.0, 50.0, 4).unwrap();
        assert_relative_eq!(scales.as_slice()[0], 2.5);
        assert_relative_eq!(scales.as_slice()[1], 100.0 / 30.0);
        assert_relative_eq!(scales.as_slice()[2], 5.0);
        assert_relative_eq!(scales.as_slice()[3], 10.0);
    }

    #[test]
    fn converts_scales_to_frequencies() {
        let scales = Scales::from_scales(100, vec![2.0, 4.0, 10.0]).unwrap();
        assert_eq!(scales.frequencies(), vec![50.0, 25.0, 10.0]);
    }

    #[test]
    fn rejects_above_nyquist() {
        let err = Scales::new(ScaleType::LinearScales, 100, 10.0, 60.0, 4).unwrap_err();
        assert_eq!(
            err,
            ScaleError::FrequencyAboveNyquist {
                frequency: 60.0,
                sample_rate: 100
            }
        );
    }
}
