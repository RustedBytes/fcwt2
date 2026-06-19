use crate::discrete::{
    TransformError, circular_downsample, circular_upsample, validate_power_of_two,
};
use crate::{DiscreteWavelet, WaveletFilterBank};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PacketBand {
    Approximation,
    Detail,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WaveletPacketNode {
    pub path: Vec<PacketBand>,
    pub coefficients: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WaveletPacketTree {
    levels: usize,
    filter_bank: WaveletFilterBank,
    leaves: Vec<WaveletPacketNode>,
}

impl WaveletPacketTree {
    pub fn levels(&self) -> usize {
        self.levels
    }

    pub fn leaves(&self) -> &[WaveletPacketNode] {
        &self.leaves
    }

    pub fn into_leaves(self) -> Vec<WaveletPacketNode> {
        self.leaves
    }

    pub fn from_leaves(
        levels: usize,
        leaves: Vec<WaveletPacketNode>,
        wavelet: DiscreteWavelet,
    ) -> Result<Self, TransformError> {
        Ok(Self {
            levels,
            filter_bank: wavelet.filter_bank()?,
            leaves,
        })
    }

    pub fn from_leaves_with_filter_bank(
        levels: usize,
        leaves: Vec<WaveletPacketNode>,
        filter_bank: WaveletFilterBank,
    ) -> Self {
        Self {
            levels,
            filter_bank,
            leaves,
        }
    }

    pub fn filter_bank(&self) -> &WaveletFilterBank {
        &self.filter_bank
    }

    pub fn reconstruct(&self) -> Result<Vec<f32>, TransformError> {
        if self.levels == 0 {
            return self
                .leaves
                .first()
                .map(|node| node.coefficients.clone())
                .ok_or(TransformError::InvalidCoefficientTree);
        }

        let expected_leaves = 1usize << self.levels;
        if self.leaves.len() != expected_leaves {
            return Err(TransformError::InvalidCoefficientTree);
        }

        let mut nodes = self.leaves.clone();
        for level in (0..self.levels).rev() {
            let expected_path_len = level + 1;
            let mut parents = Vec::with_capacity(1 << level);
            for pair in nodes.chunks_exact(2) {
                let left = &pair[0];
                let right = &pair[1];
                if left.path.len() != expected_path_len
                    || right.path.len() != expected_path_len
                    || left.path[..level] != right.path[..level]
                    || left.path[level] != PacketBand::Approximation
                    || right.path[level] != PacketBand::Detail
                {
                    return Err(TransformError::InvalidCoefficientTree);
                }

                let mut path = left.path.clone();
                path.pop();
                parents.push(WaveletPacketNode {
                    path,
                    coefficients: circular_upsample(
                        &left.coefficients,
                        &right.coefficients,
                        self.filter_bank.synthesis_low(),
                        self.filter_bank.synthesis_high(),
                    )?,
                });
            }
            nodes = parents;
        }

        if nodes.len() != 1 || !nodes[0].path.is_empty() {
            return Err(TransformError::InvalidCoefficientTree);
        }

        Ok(nodes.remove(0).coefficients)
    }
}

#[derive(Clone, Debug)]
pub struct WaveletPacketTransform {
    levels: usize,
    filter_bank: WaveletFilterBank,
}

impl WaveletPacketTransform {
    pub fn new(levels: usize) -> Self {
        Self {
            levels,
            filter_bank: WaveletFilterBank::haar(),
        }
    }

    pub fn with_wavelet(levels: usize, wavelet: DiscreteWavelet) -> Result<Self, TransformError> {
        Ok(Self {
            levels,
            filter_bank: wavelet.filter_bank()?,
        })
    }

    pub fn with_filter_bank(levels: usize, filter_bank: WaveletFilterBank) -> Self {
        Self {
            levels,
            filter_bank,
        }
    }

    pub fn levels(&self) -> usize {
        self.levels
    }

    pub fn filter_bank(&self) -> &WaveletFilterBank {
        &self.filter_bank
    }

    pub fn decompose(&self, input: &[f32]) -> Result<WaveletPacketTree, TransformError> {
        validate_power_of_two(input.len(), self.levels)?;

        let mut nodes = vec![WaveletPacketNode {
            path: Vec::new(),
            coefficients: input.to_vec(),
        }];

        for _ in 0..self.levels {
            let mut next = Vec::with_capacity(nodes.len() * 2);
            for node in nodes {
                let (approx, detail) = circular_downsample(
                    &node.coefficients,
                    self.filter_bank.analysis_low(),
                    self.filter_bank.analysis_high(),
                );
                let mut approx_path = node.path.clone();
                approx_path.push(PacketBand::Approximation);
                next.push(WaveletPacketNode {
                    path: approx_path,
                    coefficients: approx,
                });

                let mut detail_path = node.path;
                detail_path.push(PacketBand::Detail);
                next.push(WaveletPacketNode {
                    path: detail_path,
                    coefficients: detail,
                });
            }
            nodes = next;
        }

        Ok(WaveletPacketTree {
            levels: self.levels,
            filter_bank: self.filter_bank.clone(),
            leaves: nodes,
        })
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::{PacketBand, WaveletPacketTransform};
    use crate::{DiscreteWavelet, TransformError};

    #[test]
    fn rejects_invalid_inputs() {
        assert_eq!(
            WaveletPacketTransform::new(1).decompose(&[]),
            Err(TransformError::EmptyInput)
        );
        assert_eq!(
            WaveletPacketTransform::new(1).decompose(&[1.0, 2.0, 3.0]),
            Err(TransformError::NonPowerOfTwo { len: 3 })
        );
        assert_eq!(
            WaveletPacketTransform::new(3).decompose(&[1.0, 2.0, 3.0, 4.0]),
            Err(TransformError::LevelTooDeep {
                levels: 3,
                max_levels: 2
            })
        );
    }

    #[test]
    fn decomposes_expected_packet_paths() {
        let tree = WaveletPacketTransform::new(2)
            .decompose(&[1.0, 2.0, 3.0, 4.0])
            .unwrap();

        assert_eq!(tree.levels(), 2);
        assert_eq!(tree.leaves().len(), 4);
        assert_eq!(
            tree.leaves()[0].path,
            vec![PacketBand::Approximation, PacketBand::Approximation]
        );
        assert_eq!(
            tree.leaves()[3].path,
            vec![PacketBand::Detail, PacketBand::Detail]
        );
        assert_eq!(tree.leaves()[0].coefficients.len(), 1);
    }

    #[test]
    fn reconstructs_input() {
        let input = [1.0, -2.0, 3.5, 4.25, -5.0, 6.0, 7.0, -8.0];
        let tree = WaveletPacketTransform::new(3).decompose(&input).unwrap();
        let reconstructed = tree.reconstruct().unwrap();

        for (actual, expected) in reconstructed.iter().zip(input) {
            assert_relative_eq!(*actual, expected, epsilon = 1e-5);
        }
    }

    #[test]
    fn reconstructs_input_with_supported_wavelets() {
        let input = [
            1.0, -2.0, 3.5, 4.25, -5.0, 6.0, 7.0, -8.0, 0.5, 2.25, -1.75, 9.0, 3.0, -4.0, 5.5, -6.5,
        ];
        let wavelets = [
            DiscreteWavelet::Daubechies(2),
            DiscreteWavelet::Daubechies(4),
            DiscreteWavelet::Daubechies(6),
            DiscreteWavelet::Daubechies(8),
            DiscreteWavelet::Symlet(2),
            DiscreteWavelet::Symlet(4),
            DiscreteWavelet::Symlet(6),
            DiscreteWavelet::Symlet(8),
        ];

        for wavelet in wavelets {
            let transform = WaveletPacketTransform::with_wavelet(2, wavelet).unwrap();
            let tree = transform.decompose(&input).unwrap();
            let reconstructed = tree.reconstruct().unwrap();

            for (actual, expected) in reconstructed.iter().zip(input) {
                assert_relative_eq!(*actual, expected, epsilon = 5e-5);
            }
        }
    }
}
