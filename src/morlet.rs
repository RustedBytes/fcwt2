use rustfft::num_complex::Complex32;

use crate::{IPI4, PI, Wavelet};

#[derive(Clone, Debug)]
pub struct Morlet {
    bandwidth: f32,
    inverse_bandwidth: f32,
    bandwidth_squared_twice: f32,
    width: usize,
    mother: Vec<f32>,
}

impl Morlet {
    pub fn new(bandwidth: f32) -> Self {
        assert!(
            bandwidth.is_finite() && bandwidth > 0.0,
            "Morlet bandwidth must be finite and greater than zero"
        );

        Self {
            bandwidth,
            inverse_bandwidth: 1.0 / bandwidth,
            bandwidth_squared_twice: 2.0 * bandwidth * bandwidth,
            width: 0,
            mother: Vec::new(),
        }
    }

    pub fn bandwidth(&self) -> f32 {
        self.bandwidth
    }

    pub fn width(&self) -> usize {
        self.width
    }
}

impl Wavelet for Morlet {
    fn generate_frequency(&mut self, size: usize) {
        self.width = size;
        let to_radians = 2.0 * PI / size as f32;
        let norm = (2.0 * PI).sqrt() * IPI4;

        self.mother.resize(self.width, 0.0);
        for (w, mother) in self.mother.iter_mut().enumerate() {
            let tmp = 2.0 * (w as f32 * to_radians) * self.bandwidth - 2.0 * PI * self.bandwidth;
            *mother = norm * (-(tmp * tmp) / 2.0).exp();
        }
    }

    fn generate_time(&mut self, size: usize, scale: f32) -> Vec<Complex32> {
        self.width = self.support(scale);
        let len = self.width * 2 + 1;
        let norm = size as f32 * self.inverse_bandwidth * IPI4;

        (0..len)
            .map(|t| {
                let tmp1 = (t as isize - self.width as isize) as f32 / scale;
                let tmp2 = (-tmp1 * tmp1 / self.bandwidth_squared_twice).exp();
                Complex32::new(
                    norm * tmp2 * (tmp1 * 2.0 * PI).cos() / scale,
                    norm * tmp2 * (tmp1 * 2.0 * PI).sin() / scale,
                )
            })
            .collect()
    }

    fn support(&self, scale: f32) -> usize {
        (self.bandwidth * scale * 3.0) as usize
    }

    fn mother(&self) -> &[f32] {
        &self.mother
    }
}

#[cfg(test)]
mod tests {
    use super::Morlet;
    use crate::Wavelet;

    #[test]
    fn reports_support_like_cpp() {
        let morlet = Morlet::new(2.0);
        assert_eq!(morlet.support(10.0), 60);
    }

    #[test]
    fn generated_wavelet_has_support_width_on_each_side() {
        let mut morlet = Morlet::new(2.0);
        let wav = morlet.wavelet(10.0, 128);
        assert_eq!(wav.len(), 121);
        assert_eq!(morlet.width(), 60);
    }

    #[test]
    fn frequency_generation_fills_mother() {
        let mut morlet = Morlet::new(2.0);
        morlet.generate_frequency(64);
        assert_eq!(morlet.mother().len(), 64);
        assert!(morlet.mother().iter().any(|value| *value > 0.0));
    }

    #[test]
    #[should_panic(expected = "Morlet bandwidth must be finite and greater than zero")]
    fn rejects_invalid_bandwidth() {
        let _ = Morlet::new(0.0);
    }
}
