use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyComplex, PyList},
};
use rustfft::num_complex::Complex32;

use crate::{Fcwt, Morlet, ScaleError, ScaleType, Scales};

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
}

#[pyclass(name = "Scales", skip_from_py_object)]
#[derive(Clone)]
struct PyScales {
    inner: Scales,
}

#[pymethods]
impl PyScales {
    #[staticmethod]
    fn linear_scales(sample_rate: usize, f0: f32, f1: f32, nscales: usize) -> PyResult<Self> {
        Self::build(ScaleType::LinearScales, sample_rate, f0, f1, nscales)
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

    fn __len__(&self) -> usize {
        self.inner.len()
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
}

#[pymethods]
impl PyFcwt {
    #[new]
    fn new(wavelet: PyRef<'_, PyMorlet>) -> Self {
        Self {
            inner: Fcwt::new(wavelet.inner.clone()),
        }
    }

    #[staticmethod]
    fn morlet(bandwidth: f32) -> PyResult<Self> {
        validate_bandwidth(bandwidth)?;
        Ok(Self {
            inner: Fcwt::new(Morlet::new(bandwidth)),
        })
    }

    fn with_normalization(&mut self, normalize: bool) {
        let wavelet = self.inner.wavelet().clone();
        self.inner = Fcwt::new(wavelet).with_normalization(normalize);
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
        input: Vec<(f32, f32)>,
        scales: PyRef<'_, PyScales>,
    ) -> PyResult<Bound<'py, PyList>> {
        let input = input
            .into_iter()
            .map(|(re, im)| Complex32::new(re, im))
            .collect::<Vec<_>>();
        complex_list(py, self.inner.cwt_complex(&input, &scales.inner))
    }
}

fn complex_list<'py>(py: Python<'py>, values: Vec<Complex32>) -> PyResult<Bound<'py, PyList>> {
    let values = values
        .into_iter()
        .map(|value| PyComplex::from_doubles(py, value.re as f64, value.im as f64))
        .collect::<Vec<_>>();
    PyList::new(py, values)
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

fn validate_bandwidth(bandwidth: f32) -> PyResult<()> {
    if bandwidth.is_finite() && bandwidth > 0.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(
            "Morlet bandwidth must be finite and greater than zero",
        ))
    }
}

#[pymodule]
fn fcwt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMorlet>()?;
    m.add_class::<PyScales>()?;
    m.add_class::<PyFcwt>()?;
    Ok(())
}
