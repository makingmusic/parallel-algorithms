use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
fn rust_parallel_sort(py: Python<'_>, mut input: Vec<i64>) -> PyResult<Vec<i64>> {
    // Release the GIL during parallel sort
    py.allow_threads(|| {
        input.par_sort_unstable();
    });
    Ok(input)
}

#[pymodule]
fn rust_parallel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_parallel_sort, m)?)?;
    Ok(())
}


