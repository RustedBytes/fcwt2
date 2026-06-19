use fcwt2::{
    BasisSelectionCriterion, DiscreteWavelet, PacketBand, TransformError, TransformKind,
    WaveletPacketTransform, WaveletPacketTree, select_basis,
};

fn main() -> Result<(), TransformError> {
    let n = 256;
    let levels = 4;
    let clean = synthetic_signal(n);
    let noisy = add_deterministic_noise(&clean, 0.20);
    let candidates = [
        DiscreteWavelet::Haar,
        DiscreteWavelet::Daubechies(2),
        DiscreteWavelet::Daubechies(4),
        DiscreteWavelet::Daubechies(8),
        DiscreteWavelet::Symlet(4),
        DiscreteWavelet::Symlet(8),
    ];

    let selected = select_basis(
        &noisy,
        levels,
        &candidates,
        TransformKind::WaveletPacket,
        BasisSelectionCriterion::CoarsestDetailPdfShape,
    )?;

    let transform = WaveletPacketTransform::with_wavelet(levels, selected.selected)?;
    let tree = transform.decompose(&noisy)?;
    let threshold = 0.45 * universal_threshold(&tree, noisy.len());
    let denoised = threshold_packet_tree(tree, selected.selected, threshold)?;

    println!("selected basis: {}", selected.selected_name);
    println!("threshold:      {threshold:.4}");
    println!("noisy RMSE:     {:.4}", rmse(&clean, &noisy));
    println!("denoised RMSE:  {:.4}", rmse(&clean, &denoised));

    Ok(())
}

fn threshold_packet_tree(
    tree: WaveletPacketTree,
    wavelet: DiscreteWavelet,
    threshold: f32,
) -> Result<Vec<f32>, TransformError> {
    let levels = tree.levels();
    let mut leaves = tree.into_leaves();
    for leaf in &mut leaves {
        if leaf.path.contains(&PacketBand::Detail) {
            for coefficient in &mut leaf.coefficients {
                *coefficient = soft_threshold(*coefficient, threshold);
            }
        }
    }

    WaveletPacketTree::from_leaves(levels, leaves, wavelet)?.reconstruct()
}

fn universal_threshold(tree: &WaveletPacketTree, input_len: usize) -> f32 {
    let mut detail_magnitudes = tree
        .leaves()
        .iter()
        .filter(|leaf| leaf.path.contains(&PacketBand::Detail))
        .flat_map(|leaf| leaf.coefficients.iter().map(|value| value.abs()))
        .collect::<Vec<_>>();

    if detail_magnitudes.is_empty() {
        return 0.0;
    }

    detail_magnitudes.sort_by(|left, right| left.total_cmp(right));
    let median = detail_magnitudes[detail_magnitudes.len() / 2];
    let sigma = median / 0.6745;
    sigma * (2.0 * (input_len as f32).ln()).sqrt()
}

fn soft_threshold(value: f32, threshold: f32) -> f32 {
    value.signum() * (value.abs() - threshold).max(0.0)
}

fn synthetic_signal(n: usize) -> Vec<f32> {
    (0..n)
        .map(|sample| {
            let t = sample as f32 / n as f32;
            let slow = (2.0 * std::f32::consts::PI * 5.0 * t).sin();
            let burst_window = (-0.5 * ((t - 0.58) / 0.05).powi(2)).exp();
            let burst = burst_window * (2.0 * std::f32::consts::PI * 42.0 * t).sin();
            let step = if t > 0.35 { 0.35 } else { 0.0 };
            0.65 * slow + 0.8 * burst + step
        })
        .collect()
}

fn add_deterministic_noise(input: &[f32], amplitude: f32) -> Vec<f32> {
    let mut state = 0x9E37_79B9_u32;
    input
        .iter()
        .map(|value| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            value + amplitude * unit
        })
        .collect()
}

fn rmse(left: &[f32], right: &[f32]) -> f32 {
    let mean_square = left
        .iter()
        .zip(right)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f32>()
        / left.len() as f32;
    mean_square.sqrt()
}
