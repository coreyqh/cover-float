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
