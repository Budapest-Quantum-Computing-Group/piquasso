/*
 * Copyright 2021-2026 Budapest Quantum Computing Group
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <complex>
#include <cstdint>
#include <vector>

#include "bs_output_state.hpp"
#include "numpy_utils.hpp"

namespace py = pybind11;

template <typename TScalar>
py::object calculate_BS_output_state_np(
    const py::array_t<TScalar, py::array::c_style> &interferometer,
    const py::array &initial_state,
    const py::array &postselect_modes,
    const py::array &postselect_photons
)
{
    if (interferometer.ndim() != 2
            || interferometer.shape(0) != interferometer.shape(1))
    {
        throw py::value_error("interferometer must be a square matrix");
    }

    const auto initial_state_vector = nonnegative_vector_from_numpy<std::int32_t>(
        initial_state, "initial_state"
    );
    const auto modes = nonnegative_vector_from_numpy<std::int32_t>(
        postselect_modes, "postselect_modes"
    );
    const auto photons = nonnegative_vector_from_numpy<std::int32_t>(
        postselect_photons, "postselect_photons"
    );

    const auto d = static_cast<py::ssize_t>(initial_state_vector.size());

    if (interferometer.shape(0) != d)
    {
        throw py::value_error("interferometer and initial_state size mismatch");
    }

    if (modes.size() != photons.size())
    {
        throw py::value_error("postselect modes and photons must match in size");
    }

    std::vector<bool> is_postselected(d, false);

    for (const std::int32_t mode : modes)
    {
        if (mode >= d)
        {
            throw py::value_error("postselect mode out of range");
        }

        if (is_postselected[mode])
        {
            throw py::value_error("postselect modes must be unique");
        }

        is_postselected[mode] = true;
    }

    const Matrix<TScalar> interferometer_matrix = numpy_to_matrix(interferometer);

    Vector<TScalar> state;

    {
        py::gil_scoped_release release;

        state = calculate_BS_output_state<TScalar>(
            interferometer_matrix, initial_state_vector, modes, photons
        );
    }

    return create_numpy_vector(state);
}

PYBIND11_MODULE(bs_output_state, module)
{
    module.def(
        "calculate_BS_output_state",
        &calculate_BS_output_state_np<std::complex<float>>,
        py::arg("interferometer"),
        py::arg("initial_state"),
        py::arg("postselect_modes"),
        py::arg("postselect_photons")
    );
    module.def(
        "calculate_BS_output_state",
        &calculate_BS_output_state_np<std::complex<double>>,
        py::arg("interferometer"),
        py::arg("initial_state"),
        py::arg("postselect_modes"),
        py::arg("postselect_photons")
    );
}
