#include <torch/extension.h>
#include <sstream>

namespace bittorch {

// Dummy function to verify the extension is working
torch::Tensor dummy(torch::Tensor input) {
  return input.clone();
}

// Forward declaration of unpacked CUDA kernel (kept for backwards compat)
torch::Tensor ternary_gemm_cuda(
    torch::Tensor X,
    torch::Tensor W_tern,
    torch::Tensor scale,
    c10::optional<torch::Tensor> bias);

// Forward declaration of PRODUCTION packed kernel (transposed layout)
namespace production {
torch::Tensor ternary_gemm_production_cuda(
    torch::Tensor X,
    torch::Tensor W_T,  // [K_bytes, N] transposed layout
    torch::Tensor scale,
    c10::optional<torch::Tensor> bias);
}  // namespace production

/**
 * Format tensor shape for error messages.
 */
static std::string format_shape(const torch::Tensor& t) {
  std::ostringstream ss;
  ss << "[";
  for (int i = 0; i < t.dim(); i++) {
    if (i > 0) ss << ", ";
    ss << t.size(i);
  }
  ss << "]";
  return ss.str();
}

/**
 * Ternary linear forward pass (unpacked weights).
 *
 * Computes Y = X @ (W_tern * scale).T + bias
 *
 * @param X Input tensor [B, K] or [*, K]
 * @param W_tern Ternary weights [N, K] as int8 ({-1, 0, +1})
 * @param scale Per-channel scale [N]
 * @param bias Optional bias [N]
 * @return Output tensor [B, N] or [*, N]
 */
torch::Tensor ternary_linear_forward(
    torch::Tensor X,
    torch::Tensor W_tern,
    torch::Tensor scale,
    c10::optional<torch::Tensor> bias) {

  TORCH_CHECK(X.dim() >= 1, "X must have at least 1 dimension");
  TORCH_CHECK(X.numel() > 0, "X is empty");
  TORCH_CHECK(W_tern.dim() == 2, "W_tern must be 2D [N, K]");
  TORCH_CHECK(W_tern.dtype() == torch::kInt8, "W_tern must be int8");
  TORCH_CHECK(scale.dim() == 1, "scale must be 1D [N]");

  auto input_shape = X.sizes().vec();
  int K = input_shape.back();
  int B = X.numel() / K;
  int N = W_tern.size(0);
  int W_K = W_tern.size(1);

  TORCH_CHECK(K == W_K, "input K doesn't match weight K");
  TORCH_CHECK(scale.size(0) == N, "scale size doesn't match N");

  if (bias.has_value()) {
    TORCH_CHECK(bias->dim() == 1 && bias->size(0) == N, "bias shape mismatch");
  }

  TORCH_CHECK(X.is_cuda() && W_tern.is_cuda() && scale.is_cuda(), "All tensors must be CUDA");

  torch::Tensor X_2d = X.contiguous().reshape({B, K});
  torch::Tensor Y_2d = ternary_gemm_cuda(
      X_2d, W_tern.contiguous(), scale.contiguous(),
      bias.has_value() ? c10::optional<torch::Tensor>(bias->contiguous()) : c10::nullopt);

  std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end() - 1);
  output_shape.push_back(N);
  return Y_2d.reshape(output_shape);
}

/**
 * Packed ternary linear forward pass with TRANSPOSED layout.
 *
 * This is the PRODUCTION API. Uses automatic kernel selection:
 * - B <= 32: small-batch kernel (optimized for inference)
 * - B > 32: large-batch kernel (optimized for training)
 *
 * Override via BITTORCH_KERNEL env var: "small", "large", or "auto"
 *
 * @param X Input tensor [B, K] or [*, K]
 * @param W_T Packed transposed weights [K_bytes, N] as uint8
 * @param scale Per-channel scale [N]
 * @param bias Optional bias [N]
 * @param in_features Original K (needed because K_bytes = ceil(K/4))
 * @return Output tensor [B, N] or [*, N]
 */
torch::Tensor ternary_linear_packed_forward(
    torch::Tensor X,
    torch::Tensor W_T,  // [K_bytes, N] transposed layout
    torch::Tensor scale,
    c10::optional<torch::Tensor> bias,
    int64_t in_features) {

  TORCH_CHECK(X.dim() >= 1, "X must have at least 1 dimension");
  TORCH_CHECK(X.numel() > 0, "X is empty");
  TORCH_CHECK(W_T.dim() == 2, "W_T must be 2D [K_bytes, N]");
  TORCH_CHECK(W_T.dtype() == torch::kUInt8, "W_T must be uint8");
  TORCH_CHECK(scale.dim() == 1, "scale must be 1D [N]");

  auto input_shape = X.sizes().vec();
  int K = input_shape.back();
  int B = X.numel() / K;
  int K_bytes = W_T.size(0);
  int N = W_T.size(1);

  TORCH_CHECK(K == in_features, "input K doesn't match in_features");
  TORCH_CHECK(K_bytes == (in_features + 3) / 4, "K_bytes doesn't match ceil(K/4)");
  TORCH_CHECK(scale.size(0) == N, "scale size doesn't match N");

  if (bias.has_value()) {
    TORCH_CHECK(bias->dim() == 1 && bias->size(0) == N, "bias shape mismatch");
  }

  TORCH_CHECK(X.is_cuda() && W_T.is_cuda() && scale.is_cuda(), "All tensors must be CUDA");

  torch::Tensor X_2d = X.contiguous().reshape({B, K});
  torch::Tensor Y_2d = production::ternary_gemm_production_cuda(
      X_2d, W_T.contiguous(), scale.contiguous(),
      bias.has_value() ? c10::optional<torch::Tensor>(bias->contiguous()) : c10::nullopt);

  std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end() - 1);
  output_shape.push_back(N);
  return Y_2d.reshape(output_shape);
}

}  // namespace bittorch
