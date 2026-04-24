/* Ryan Wolk (rwolk@g.hmc.edu)
 * This module links against the unmodified copy of softfloat. Calls into this modules have the purpose
 * of verifying that our changes to softfloat have not altered its ability to produce the correct results.
 */

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
    m.doc() = "Python bindings for the unmodified coverfloat reference model. Use these functions to verify the "
              "correctness of our changes to softfloat";

    m.def(
        "run_test_vector",
        &run_test_vector,
        R"pbdoc(
      Run the given vector through a coverfloat reference model using an unmodified copy of softfloat. Use this function to verify that the any test vector is producing the correct output.
  )pbdoc",
        py::arg("test_vector"),
        py::arg("suppress_error_check") = true
    );
}
