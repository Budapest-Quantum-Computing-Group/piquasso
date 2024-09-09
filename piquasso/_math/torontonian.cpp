#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "numpy_utils.hpp"
#include "torontonian.hpp"

namespace py = pybind11;

template <typename TScalar>
py::object torontonian_np(py::array_t<TScalar, py::array::c_style | py::array::forcecast> array)
{
    Matrix<TScalar> matrix = numpy_to_matrix(array);

    TScalar result = torontonian_cpp(matrix);

    return create_numpy_scalar(result);
}

PYBIND11_MODULE(torontonian, m)
{
    m.def("torontonian", &torontonian_np<float>);
    m.def("torontonian", &torontonian_np<double>);
}
