#![feature(portable_simd)]

pub mod basis_selection;
mod discrete;
pub mod dual_tree_complex;
mod fcwt;
mod filter_bank;
mod morlet;
#[cfg(feature = "python")]
mod python;
mod scales;
pub mod stationary;
mod wavelet;
pub mod wavelet_packet;

pub use basis_selection::{
    BasisScore, BasisSelectionCriterion, SelectedBasis, TransformKind, score_basis, select_basis,
};
pub use discrete::TransformError;
pub use dual_tree_complex::{DtcwtLevel, DtcwtTree, DualTreeComplexWaveletTransform};
pub use fcwt::Fcwt;
pub use filter_bank::{DiscreteWavelet, WaveletFilterBank};
pub use morlet::Morlet;
pub use rustfft::num_complex::Complex32;
pub use scales::{ScaleError, ScaleType, Scales};
pub use stationary::{StationaryWaveletTransform, SwtCoefficients, SwtLevel};
pub use wavelet::Wavelet;
pub use wavelet_packet::{
    PacketBand, WaveletPacketNode, WaveletPacketTransform, WaveletPacketTree,
};

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
