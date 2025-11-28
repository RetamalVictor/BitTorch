#include <torch/extension.h>
#include <sstream>

namespace bittorch {

// Dummy function to verify the extension is working
torch::Tensor dummy(torch::Tensor input) {
  // Simply return a clone of the input
  return input.clone();
}

// Forward declaration of CUDA kernel
torch::Tensor ternary_gemm_cuda(
    torch::Tensor X,
    torch::Tensor W_tern,
    torch::Tensor scale,
    c10::optional<torch::Tensor> bias);

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
 * Ternary linear forward pass.
 *
 * Computes Y = X @ (W_tern * scale).T + bias
 *
 * This function dispatches to CUDA or falls back to CPU implementation.
 *
 * @param X Input tensor [B, K] or [*, K] (will be reshaped)
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

  // ========================================
  // Input validation with helpful messages
  // ========================================

  // Check X is valid
  TORCH_CHECK(X.dim() >= 1,
              "ternary_linear_forward: X must have at least 1 dimension, got ",
              X.dim());
  TORCH_CHECK(X.numel() > 0,
              "ternary_linear_forward: X is empty (numel=0)");

  // Check W_tern is valid
  TORCH_CHECK(W_tern.dim() == 2,
              "ternary_linear_forward: W_tern must be 2D [N, K], got shape ",
              format_shape(W_tern));
  TORCH_CHECK(W_tern.dtype() == torch::kInt8,
              "ternary_linear_forward: W_tern must be int8, got ",
              W_tern.dtype());

  // Check scale is valid
  TORCH_CHECK(scale.dim() == 1,
              "ternary_linear_forward: scale must be 1D [N], got shape ",
              format_shape(scale));

  // Get dimensions
  auto input_shape = X.sizes().vec();
  int K = input_shape.back();
  int B = X.numel() / K;
  int N = W_tern.size(0);
  int W_K = W_tern.size(1);

  // Check dimension compatibility
  TORCH_CHECK(K == W_K,
              "ternary_linear_forward: input features K=", K,
              " doesn't match weight K=", W_K,
              ". X shape: ", format_shape(X),
              ", W_tern shape: ", format_shape(W_tern));

  TORCH_CHECK(scale.size(0) == N,
              "ternary_linear_forward: scale size ", scale.size(0),
              " doesn't match N=", N,
              " (output features from W_tern)");

  // Check bias if provided
  if (bias.has_value()) {
    TORCH_CHECK(bias->dim() == 1,
                "ternary_linear_forward: bias must be 1D [N], got shape ",
                format_shape(*bias));
    TORCH_CHECK(bias->size(0) == N,
                "ternary_linear_forward: bias size ", bias->size(0),
                " doesn't match N=", N);
  }

  // ========================================
  // Device checks
  // ========================================

  TORCH_CHECK(X.is_cuda(),
              "ternary_linear_forward: X must be on CUDA device, got ",
              X.device());
  TORCH_CHECK(W_tern.is_cuda(),
              "ternary_linear_forward: W_tern must be on CUDA device, got ",
              W_tern.device());
  TORCH_CHECK(scale.is_cuda(),
              "ternary_linear_forward: scale must be on CUDA device, got ",
              scale.device());

  // Check all tensors on same device
  TORCH_CHECK(X.device() == W_tern.device(),
              "ternary_linear_forward: X and W_tern must be on same device. "
              "X on ", X.device(), ", W_tern on ", W_tern.device());
  TORCH_CHECK(X.device() == scale.device(),
              "ternary_linear_forward: X and scale must be on same device. "
              "X on ", X.device(), ", scale on ", scale.device());

  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda(),
                "ternary_linear_forward: bias must be on CUDA device, got ",
                bias->device());
    TORCH_CHECK(X.device() == bias->device(),
                "ternary_linear_forward: X and bias must be on same device. "
                "X on ", X.device(), ", bias on ", bias->device());
  }

  // ========================================
  // Contiguity - make tensors contiguous
  // ========================================

  torch::Tensor X_contig = X.contiguous();
  torch::Tensor W_contig = W_tern.contiguous();
  torch::Tensor scale_contig = scale.contiguous();
  c10::optional<torch::Tensor> bias_contig;
  if (bias.has_value()) {
    bias_contig = bias->contiguous();
  }

  // ========================================
  // Execute kernel
  // ========================================

  // Reshape [*, K] -> [B, K]
  torch::Tensor X_2d = X_contig.reshape({B, K});

  torch::Tensor Y_2d = ternary_gemm_cuda(X_2d, W_contig, scale_contig, bias_contig);

  // Reshape output back to [*, N]
  std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end() - 1);
  output_shape.push_back(N);

  return Y_2d.reshape(output_shape);
}

}  // namespace bittorch
