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
    torch::Tensor W_T,  // [K_bytes, N] transposed layout
    torch::Tensor scale,
    c10::optional<torch::Tensor> bias,
    int64_t in_features);

// Tensor Core kernel
namespace tensor_core {
torch::Tensor ternary_gemm_tensor_core_cuda(
    torch::Tensor X,
    torch::Tensor W_T,
    torch::Tensor scale,
    c10::optional<torch::Tensor> bias,
    int64_t in_features,
    int64_t version);
} // namespace tensor_core

} // namespace bittorch

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "BitTorch: High-performance low-precision backend for PyTorch";

  m.def("dummy", &bittorch::dummy,
        "Dummy function to verify extension is working",
        py::arg("input"));

  m.def("ternary_linear_forward", &bittorch::ternary_linear_forward,
        "Ternary linear forward pass (unpacked weights)",
        py::arg("X"),
        py::arg("W_tern"),
        py::arg("scale"),
        py::arg("bias") = py::none());

  m.def("ternary_linear_packed_forward", &bittorch::ternary_linear_packed_forward,
        R"doc(
        Packed ternary linear forward pass with automatic kernel selection.

        Uses TRANSPOSED weight layout [K_bytes, N] for optimal memory access.
        Automatically selects kernel based on batch size:
        - B <= 32: small-batch kernel (optimized for inference)
        - B > 32: large-batch kernel (optimized for training)

        Override via BITTORCH_KERNEL env var: "small", "large", or "auto"
        )doc",
        py::arg("X"),
        py::arg("W_T"),
        py::arg("scale"),
        py::arg("bias") = py::none(),
        py::arg("in_features"));

  m.def("ternary_linear_tensor_core_forward", &bittorch::tensor_core::ternary_gemm_tensor_core_cuda,
        R"doc(
        Tensor Core ternary linear forward pass.

        Uses INT8 Tensor Cores with LUT-based 2-bit to INT8 unpacking.

        Args:
            X: Input tensor [B, K] float32
            W_T: Packed transposed weights [K_bytes, N] uint8
            scale: Per-channel scale [N] float32
            bias: Optional bias [N] float32
            in_features: Number of input features (K)
            version: Kernel version (1=V1 float compute, 2=V2 mma.sync)

        Returns:
            Output tensor [B, N] float32
        )doc",
        py::arg("X"),
        py::arg("W_T"),
        py::arg("scale"),
        py::arg("bias") = py::none(),
        py::arg("in_features"),
        py::arg("version") = 1);
}
