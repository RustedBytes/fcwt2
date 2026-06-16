#![feature(portable_simd)]

mod fcwt;
mod morlet;
mod scales;
mod wavelet;

pub use fcwt::Fcwt;
pub use morlet::Morlet;
pub use rustfft::num_complex::Complex32;
pub use scales::{ScaleError, ScaleType, Scales};
pub use wavelet::Wavelet;

const PI: f32 = std::f32::consts::PI;
const IPI4: f32 = 0.751_125_6_f32;

pub(crate) fn next_power_of_two_len(n: usize) -> usize {
    if n <= 1 { 1 } else { n.next_power_of_two() }
}

#[cfg(test)]
mod tests {
    use super::next_power_of_two_len;

    #[test]
    fn finds_next_power_of_two() {
        assert_eq!(next_power_of_two_len(0), 1);
        assert_eq!(next_power_of_two_len(1), 1);
        assert_eq!(next_power_of_two_len(2), 2);
        assert_eq!(next_power_of_two_len(3), 4);
        assert_eq!(next_power_of_two_len(1025), 2048);
    }
}
