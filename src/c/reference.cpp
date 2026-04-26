// Copyright (C) 2025-26 Harvey Mudd College
//
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
//
// Licensed under the Solderpad Hardware License v 2.1 (the “License”); you may not use this file
// except in compliance with the License, or, at your option, the Apache License version 2.0. You
// may obtain a copy of the License at
//
// https://solderpad.org/licenses/SHL-2.1/
//
// Unless required by applicable law or agreed to in writing, any work distributed under the
// License is distributed on an “AS IS” BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions
// and limitations under the License.

#include <coverfloat.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;

std::string run_test_vector(const std::string &test_vector, bool suppress_error_check = true) {
    std::string res = coverfloat_runtestvector(test_vector, suppress_error_check);

    if (res.size() != COVER_VECTOR_WIDTH_HEX_WITH_SEPARATORS + 1) {
        throw py::value_error("Error running test vector: " + test_vector + "\nModel Information: " + res);
    }

    return res;
}

PYBIND11_MODULE(_reference, m) {
    m.doc() = "Python bindings for the coverfloat reference model, providing functions to run test vectors.";

    m.def(
        "run_test_vector",
        &run_test_vector,
        R"pbdoc(
      Run the given vector through the coverfloat reference model.
  )pbdoc",
        py::arg("test_vector"),
        py::arg("suppress_error_check") = true
    );
}
