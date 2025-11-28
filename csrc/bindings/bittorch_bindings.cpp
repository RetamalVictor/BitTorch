#include <torch/extension.h>

namespace bittorch {

// Forward declarations
torch::Tensor dummy(torch::Tensor input);

torch::Tensor ternary_linear_forward(
    torch::Tensor X,
    torch::Tensor W_tern,
    torch::Tensor scale,
    c10::optional<torch::Tensor> bias);

torch::Tensor ternary_linear_packed_forward(
    torch::Tensor X,
    torch::Tensor W_packed,
    torch::Tensor scale,
    c10::optional<torch::Tensor> bias,
    int64_t in_features);

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

  m.def("ternary_linear_packed_forward", &bittorch::ternary_linear_packed_forward,
        "Packed ternary linear forward pass (CUDA) - reads 2-bit packed weights directly",
        py::arg("X"),
        py::arg("W_packed"),
        py::arg("scale"),
        py::arg("bias") = py::none(),
        py::arg("in_features"));
}
