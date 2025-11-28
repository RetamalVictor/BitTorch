#include <torch/extension.h>

namespace bittorch {

// Forward declarations
torch::Tensor dummy(torch::Tensor input);

} // namespace bittorch

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "BitTorch: High-performance low-precision backend for PyTorch";

  m.def("dummy", &bittorch::dummy,
        "Dummy function to verify extension is working",
        py::arg("input"));
}
