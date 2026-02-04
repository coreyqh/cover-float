#include <pybind11/pybind11.h>
#include <coverfloat.h>

namespace py = pybind11;

std::string run_test_vector(const std::string &test_vector, bool suppress_error_check = true) {
  const size_t OUTPUT_SIZE = 512;
  char* output = new char[OUTPUT_SIZE];

  int res = coverfloat_runtestvector(
      test_vector.c_str(),
      test_vector.size(),
      output,
      OUTPUT_SIZE,
      suppress_error_check
  );

  std::string result (output);
  delete[] output;

  if (res != EXIT_SUCCESS) {
      throw py::value_error("Error running test vector: " + result);
  }

  return result;
}

PYBIND11_MODULE(_reference, m) {
  m.doc() = "Python bindings for the coverfloat reference model, providing functions to run test vectors.";

  m.def("run_test_vector", &run_test_vector, R"pbdoc(
      Run the given vector through the coverfloat reference model.
  )pbdoc", py::arg("test_vector"), py::arg("suppress_error_check") = true);
}
