#include <torch/extension.h>

namespace bittorch {

// Forward declarations
torch::Tensor dummy(torch::Tensor input);

torch::Tensor ternary_linear_forward(
    torch::Tensor X,
    torch::Tensor W_tern,
    torch::Tensor scale,
    c10::optional<torch::Tensor> bias);

} // namespace bittorch

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "BitTorch: High-performance low-precision backend for PyTorch";

  m.def("dummy", &bittorch::dummy,
        "Dummy function to verify extension is working",
        py::arg("input"));

  m.def("ternary_linear_forward", &bittorch::ternary_linear_forward,
        "Ternary linear forward pass (CUDA)",
        py::arg("X"),
        py::arg("W_tern"),
        py::arg("scale"),
        py::arg("bias") = py::none());
}
