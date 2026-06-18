use pyo3::{
    exceptions::{PyIndexError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyComplex, PyIterator, PyList},
};
use rustfft::num_complex::Complex32;

use crate::{
    DtcwtLevel, DtcwtTree, DualTreeComplexWaveletTransform, Fcwt, Morlet, PacketBand, ScaleError,
    ScaleType, Scales, StationaryWaveletTransform, SwtCoefficients, SwtLevel, TransformError,
    WaveletPacketNode, WaveletPacketTransform, WaveletPacketTree,
};

#[pyclass(name = "Morlet", skip_from_py_object)]
#[derive(Clone)]
struct PyMorlet {
    inner: Morlet,
}

#[pymethods]
impl PyMorlet {
    #[new]
    fn new(bandwidth: f32) -> PyResult<Self> {
        validate_bandwidth(bandwidth)?;
        Ok(Self {
            inner: Morlet::new(bandwidth),
        })
    }

    #[getter]
    fn bandwidth(&self) -> f32 {
        self.inner.bandwidth()
    }

    fn __repr__(&self) -> String {
        format!("Morlet(bandwidth={})", self.inner.bandwidth())
    }
}

#[pyclass(name = "Scales", skip_from_py_object)]
#[derive(Clone)]
struct PyScales {
    inner: Scales,
}

#[pymethods]
impl PyScales {
    #[staticmethod]
    fn linear(sample_rate: usize, f0: f32, f1: f32, nscales: usize) -> PyResult<Self> {
        Self::build(ScaleType::LinearScales, sample_rate, f0, f1, nscales)
    }

    #[staticmethod]
    fn linear_scales(sample_rate: usize, f0: f32, f1: f32, nscales: usize) -> PyResult<Self> {
        Self::build(ScaleType::LinearScales, sample_rate, f0, f1, nscales)
    }

    #[staticmethod]
    #[pyo3(name = "log")]
    fn log(sample_rate: usize, f0: f32, f1: f32, nscales: usize) -> PyResult<Self> {
        Self::build(ScaleType::LogScales, sample_rate, f0, f1, nscales)
    }

    #[staticmethod]
    fn log_scales(sample_rate: usize, f0: f32, f1: f32, nscales: usize) -> PyResult<Self> {
        Self::build(ScaleType::LogScales, sample_rate, f0, f1, nscales)
    }

    #[staticmethod]
    fn linear_frequencies(sample_rate: usize, f0: f32, f1: f32, nscales: usize) -> PyResult<Self> {
        Self::build(ScaleType::LinearFrequencies, sample_rate, f0, f1, nscales)
    }

    #[staticmethod]
    fn from_scales(sample_rate: usize, scales: Vec<f32>) -> PyResult<Self> {
        Scales::from_scales(sample_rate, scales)
            .map(|inner| Self { inner })
            .map_err(scale_error)
    }

    fn frequencies(&self) -> Vec<f32> {
        self.inner.frequencies()
    }

    fn values(&self) -> Vec<f32> {
        self.inner.as_slice().to_vec()
    }

    #[getter]
    fn sample_rate(&self) -> usize {
        self.inner.sample_rate()
    }

    fn __getitem__(&self, index: isize) -> PyResult<f32> {
        let len = self.inner.len() as isize;
        let index = if index < 0 { len + index } else { index };

        if index < 0 || index >= len {
            return Err(PyIndexError::new_err("scale index out of range"));
        }

        Ok(self.inner.as_slice()[index as usize])
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        PyList::new(py, self.inner.as_slice())?.call_method0("__iter__")
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Scales(sample_rate={}, values={:?})",
            self.inner.sample_rate(),
            self.inner.as_slice()
        )
    }
}

impl PyScales {
    fn build(
        scale_type: ScaleType,
        sample_rate: usize,
        f0: f32,
        f1: f32,
        nscales: usize,
    ) -> PyResult<Self> {
        Scales::new(scale_type, sample_rate, f0, f1, nscales)
            .map(|inner| Self { inner })
            .map_err(scale_error)
    }
}

#[pyclass(name = "Fcwt")]
struct PyFcwt {
    inner: Fcwt<Morlet>,
    bandwidth: f32,
}

#[pymethods]
impl PyFcwt {
    #[new]
    #[pyo3(signature = (wavelet=None, *, normalize=true))]
    fn new(wavelet: Option<&Bound<'_, PyAny>>, normalize: bool) -> PyResult<Self> {
        let wavelet = match wavelet {
            Some(value) => parse_wavelet(value)?,
            None => Morlet::new(2.0),
        };
        Ok(Self::from_wavelet(wavelet, normalize))
    }

    #[staticmethod]
    #[pyo3(signature = (bandwidth, *, normalize=true))]
    fn morlet(bandwidth: f32, normalize: bool) -> PyResult<Self> {
        validate_bandwidth(bandwidth)?;
        Ok(Self::from_wavelet(Morlet::new(bandwidth), normalize))
    }

    fn with_normalization(&mut self, normalize: bool) {
        self.inner.set_normalization(normalize);
    }

    #[getter]
    fn normalization(&self) -> bool {
        self.inner.normalization()
    }

    #[getter]
    fn normalize(&self) -> bool {
        self.inner.normalization()
    }

    #[setter]
    fn set_normalization(&mut self, normalize: bool) {
        self.inner.set_normalization(normalize);
    }

    #[setter]
    fn set_normalize(&mut self, normalize: bool) {
        self.inner.set_normalization(normalize);
    }

    #[getter]
    fn bandwidth(&self) -> f32 {
        self.bandwidth
    }

    fn cwt_real<'py>(
        &mut self,
        py: Python<'py>,
        input: Vec<f32>,
        scales: PyRef<'_, PyScales>,
    ) -> PyResult<Bound<'py, PyList>> {
        complex_list(py, self.inner.cwt_real(&input, &scales.inner))
    }

    fn cwt_complex<'py>(
        &mut self,
        py: Python<'py>,
        input: &Bound<'_, PyAny>,
        scales: PyRef<'_, PyScales>,
    ) -> PyResult<Bound<'py, PyList>> {
        let input = complex_input(input)?;
        complex_list(py, self.inner.cwt_complex(&input, &scales.inner))
    }

    fn __repr__(&self) -> String {
        format!(
            "Fcwt(bandwidth={}, normalize={})",
            self.bandwidth,
            py_bool(self.inner.normalization())
        )
    }
}

impl PyFcwt {
    fn from_wavelet(wavelet: Morlet, normalize: bool) -> Self {
        let bandwidth = wavelet.bandwidth();
        Self {
            inner: Fcwt::new(wavelet).with_normalization(normalize),
            bandwidth,
        }
    }
}

#[pyclass(name = "WaveletPacketTransform", skip_from_py_object)]
#[derive(Clone)]
struct PyWaveletPacketTransform {
    inner: WaveletPacketTransform,
}

#[pymethods]
impl PyWaveletPacketTransform {
    #[new]
    fn new(levels: usize) -> Self {
        Self {
            inner: WaveletPacketTransform::new(levels),
        }
    }

    #[getter]
    fn levels(&self) -> usize {
        self.inner.levels()
    }

    fn decompose(&self, input: Vec<f32>) -> PyResult<PyWaveletPacketTree> {
        self.inner
            .decompose(&input)
            .map(|inner| PyWaveletPacketTree { inner })
            .map_err(transform_error)
    }

    fn __repr__(&self) -> String {
        format!("WaveletPacketTransform(levels={})", self.inner.levels())
    }
}

#[pyclass(name = "WaveletPacketTree", skip_from_py_object)]
#[derive(Clone)]
struct PyWaveletPacketTree {
    inner: WaveletPacketTree,
}

#[pymethods]
impl PyWaveletPacketTree {
    #[getter]
    fn levels(&self) -> usize {
        self.inner.levels()
    }

    fn leaves(&self) -> Vec<PyWaveletPacketNode> {
        self.inner
            .leaves()
            .iter()
            .cloned()
            .map(|inner| PyWaveletPacketNode { inner })
            .collect()
    }

    fn reconstruct(&self) -> PyResult<Vec<f32>> {
        self.inner.reconstruct().map_err(transform_error)
    }

    fn __repr__(&self) -> String {
        format!(
            "WaveletPacketTree(levels={}, leaves={})",
            self.inner.levels(),
            self.inner.leaves().len()
        )
    }
}

#[pyclass(name = "WaveletPacketNode", skip_from_py_object)]
#[derive(Clone)]
struct PyWaveletPacketNode {
    inner: WaveletPacketNode,
}

#[pymethods]
impl PyWaveletPacketNode {
    #[getter]
    fn path(&self) -> Vec<&'static str> {
        self.inner
            .path
            .iter()
            .map(|band| match band {
                PacketBand::Approximation => "a",
                PacketBand::Detail => "d",
            })
            .collect()
    }

    #[getter]
    fn coefficients(&self) -> Vec<f32> {
        self.inner.coefficients.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "WaveletPacketNode(path={:?}, coefficients={})",
            self.path(),
            self.inner.coefficients.len()
        )
    }
}

#[pyclass(name = "StationaryWaveletTransform", skip_from_py_object)]
#[derive(Clone)]
struct PyStationaryWaveletTransform {
    inner: StationaryWaveletTransform,
}

#[pymethods]
impl PyStationaryWaveletTransform {
    #[new]
    fn new(levels: usize) -> Self {
        Self {
            inner: StationaryWaveletTransform::new(levels),
        }
    }

    #[getter]
    fn levels(&self) -> usize {
        self.inner.levels()
    }

    fn decompose(&self, input: Vec<f32>) -> PyResult<PySwtCoefficients> {
        self.inner
            .decompose(&input)
            .map(|inner| PySwtCoefficients { inner })
            .map_err(transform_error)
    }

    fn reconstruct(&self, coefficients: PyRef<'_, PySwtCoefficients>) -> PyResult<Vec<f32>> {
        self.inner
            .reconstruct(&coefficients.inner)
            .map_err(transform_error)
    }

    fn __repr__(&self) -> String {
        format!("StationaryWaveletTransform(levels={})", self.inner.levels())
    }
}

#[pyclass(name = "SwtCoefficients", skip_from_py_object)]
#[derive(Clone)]
struct PySwtCoefficients {
    inner: SwtCoefficients,
}

#[pymethods]
impl PySwtCoefficients {
    #[getter]
    fn input_len(&self) -> usize {
        self.inner.input_len()
    }

    fn levels(&self) -> Vec<PySwtLevel> {
        self.inner
            .levels()
            .iter()
            .cloned()
            .map(|inner| PySwtLevel { inner })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "SwtCoefficients(input_len={}, levels={})",
            self.inner.input_len(),
            self.inner.levels().len()
        )
    }
}

#[pyclass(name = "SwtLevel", skip_from_py_object)]
#[derive(Clone)]
struct PySwtLevel {
    inner: SwtLevel,
}

#[pymethods]
impl PySwtLevel {
    #[getter]
    fn approximation(&self) -> Vec<f32> {
        self.inner.approximation.clone()
    }

    #[getter]
    fn detail(&self) -> Vec<f32> {
        self.inner.detail.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "SwtLevel(approximation={}, detail={})",
            self.inner.approximation.len(),
            self.inner.detail.len()
        )
    }
}

#[pyclass(name = "DualTreeComplexWaveletTransform", skip_from_py_object)]
#[derive(Clone)]
struct PyDualTreeComplexWaveletTransform {
    inner: DualTreeComplexWaveletTransform,
}

#[pymethods]
impl PyDualTreeComplexWaveletTransform {
    #[new]
    fn new(levels: usize) -> Self {
        Self {
            inner: DualTreeComplexWaveletTransform::new(levels),
        }
    }

    #[getter]
    fn levels(&self) -> usize {
        self.inner.levels()
    }

    fn decompose(&self, input: Vec<f32>) -> PyResult<PyDtcwtTree> {
        self.inner
            .decompose(&input)
            .map(|inner| PyDtcwtTree { inner })
            .map_err(transform_error)
    }

    fn reconstruct(&self, tree: PyRef<'_, PyDtcwtTree>) -> PyResult<Vec<f32>> {
        self.inner.reconstruct(&tree.inner).map_err(transform_error)
    }

    fn __repr__(&self) -> String {
        format!(
            "DualTreeComplexWaveletTransform(levels={})",
            self.inner.levels()
        )
    }
}

#[pyclass(name = "DtcwtTree", skip_from_py_object)]
#[derive(Clone)]
struct PyDtcwtTree {
    inner: DtcwtTree,
}

#[pymethods]
impl PyDtcwtTree {
    #[getter]
    fn lowpass(&self) -> Vec<f32> {
        self.inner.lowpass().to_vec()
    }

    fn highpasses(&self) -> Vec<PyDtcwtLevel> {
        self.inner
            .highpasses()
            .iter()
            .cloned()
            .map(|inner| PyDtcwtLevel { inner })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "DtcwtTree(lowpass={}, highpasses={})",
            self.inner.lowpass().len(),
            self.inner.highpasses().len()
        )
    }
}

#[pyclass(name = "DtcwtLevel", skip_from_py_object)]
#[derive(Clone)]
struct PyDtcwtLevel {
    inner: DtcwtLevel,
}

#[pymethods]
impl PyDtcwtLevel {
    #[getter]
    fn detail<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        complex_list(py, self.inner.detail.clone())
    }

    fn __repr__(&self) -> String {
        format!("DtcwtLevel(detail={})", self.inner.detail.len())
    }
}

fn complex_list<'py>(py: Python<'py>, values: Vec<Complex32>) -> PyResult<Bound<'py, PyList>> {
    let values = values
        .into_iter()
        .map(|value| PyComplex::from_doubles(py, value.re as f64, value.im as f64))
        .collect::<Vec<_>>();
    PyList::new(py, values)
}

fn complex_input(input: &Bound<'_, PyAny>) -> PyResult<Vec<Complex32>> {
    PyIterator::from_object(input)?
        .map(|item| {
            let item = item?;
            if let Ok(value) = item.extract::<Complex32>() {
                return Ok(value);
            }
            if let Ok((re, im)) = item.extract::<(f32, f32)>() {
                return Ok(Complex32::new(re, im));
            }
            Err(PyTypeError::new_err(
                "complex input values must be complex numbers or (real, imag) pairs",
            ))
        })
        .collect()
}

fn scale_error(error: ScaleError) -> PyErr {
    match error {
        ScaleError::FrequencyAboveNyquist {
            frequency,
            sample_rate,
        } => PyValueError::new_err(format!(
            "frequency {frequency} is above Nyquist frequency {}",
            sample_rate as f32 / 2.0
        )),
        ScaleError::EmptyScaleSet => PyValueError::new_err("scale set cannot be empty"),
        ScaleError::InvalidFrequencyRange { f0, f1 } => PyValueError::new_err(format!(
            "expected finite positive frequencies with f0 <= f1, got f0={f0}, f1={f1}"
        )),
        ScaleError::InvalidSampleRate => {
            PyValueError::new_err("sample_rate must be greater than zero")
        }
    }
}

fn transform_error(error: TransformError) -> PyErr {
    match error {
        TransformError::EmptyInput => PyValueError::new_err("input cannot be empty"),
        TransformError::NonPowerOfTwo { len } => {
            PyValueError::new_err(format!("input length {len} is not a power of two"))
        }
        TransformError::LevelTooDeep { levels, max_levels } => PyValueError::new_err(format!(
            "levels {levels} exceeds maximum decomposition level {max_levels}"
        )),
        TransformError::InvalidCoefficientTree => PyValueError::new_err("invalid coefficient tree"),
    }
}

fn validate_bandwidth(bandwidth: f32) -> PyResult<()> {
    if bandwidth.is_finite() && bandwidth > 0.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(
            "Morlet bandwidth must be finite and greater than zero",
        ))
    }
}

fn parse_wavelet(value: &Bound<'_, PyAny>) -> PyResult<Morlet> {
    if let Ok(wavelet) = value.extract::<PyRef<'_, PyMorlet>>() {
        return Ok(wavelet.inner.clone());
    }

    if let Ok(bandwidth) = value.extract::<f32>() {
        validate_bandwidth(bandwidth)?;
        return Ok(Morlet::new(bandwidth));
    }

    Err(PyTypeError::new_err(
        "expected a Morlet instance, a numeric bandwidth, or None",
    ))
}

fn py_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}

#[pymodule]
fn fcwt2(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMorlet>()?;
    m.add_class::<PyScales>()?;
    m.add_class::<PyFcwt>()?;
    m.add_class::<PyWaveletPacketTransform>()?;
    m.add_class::<PyWaveletPacketTree>()?;
    m.add_class::<PyWaveletPacketNode>()?;
    m.add_class::<PyStationaryWaveletTransform>()?;
    m.add_class::<PySwtCoefficients>()?;
    m.add_class::<PySwtLevel>()?;
    m.add_class::<PyDualTreeComplexWaveletTransform>()?;
    m.add_class::<PyDtcwtTree>()?;
    m.add_class::<PyDtcwtLevel>()?;
    m.add("FCWT", m.getattr("Fcwt")?)?;
    Ok(())
}
