use rustfft::num_complex::Complex32;

pub trait Wavelet {
    fn generate_frequency(&mut self, size: usize);

    fn generate_time(&mut self, size: usize, scale: f32) -> Vec<Complex32>;

    fn support(&self, scale: f32) -> usize;

    fn wavelet(&mut self, scale: f32, size: usize) -> Vec<Complex32> {
        self.generate_time(size, scale)
    }

    fn mother(&self) -> &[f32];

    fn imag_frequency(&self) -> bool {
        false
    }

    fn double_sided(&self) -> bool {
        false
    }
}
