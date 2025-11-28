#include <torch/extension.h>
#include <sstream>

namespace bittorch {

// Dummy function to verify the extension is working
torch::Tensor dummy(torch::Tensor input) {
  // Simply return a clone of the input
  return input.clone();
}

// Forward declaration of CUDA kernels
torch::Tensor ternary_gemm_cuda(
    torch::Tensor X,
    torch::Tensor W_tern,
    torch::Tensor scale,
    c10::optional<torch::Tensor> bias);

// Forward declaration of packed CUDA kernel
torch::Tensor ternary_gemm_packed_cuda(
    torch::Tensor X,
    torch::Tensor W_packed,
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

/**
 * Packed ternary linear forward pass.
 *
 * Computes Y = X @ (W_packed * scale).T + bias
 * where W_packed contains 2-bit ternary weights (4 per byte).
 *
 * This function dispatches to CUDA kernel that reads packed weights
 * directly without materializing full float/int8 weight tensors.
 *
 * @param X Input tensor [B, K] or [*, K] (will be reshaped)
 * @param W_packed Packed ternary weights [N, K_bytes] as uint8
 * @param scale Per-channel scale [N]
 * @param bias Optional bias [N]
 * @param in_features Original K (unpacked) - needed because K_bytes = ceil(K/4)
 * @return Output tensor [B, N] or [*, N]
 */
torch::Tensor ternary_linear_packed_forward(
    torch::Tensor X,
    torch::Tensor W_packed,
    torch::Tensor scale,
    c10::optional<torch::Tensor> bias,
    int64_t in_features) {

  // ========================================
  // Input validation with helpful messages
  // ========================================

  // Check X is valid
  TORCH_CHECK(X.dim() >= 1,
              "ternary_linear_packed_forward: X must have at least 1 dimension, got ",
              X.dim());
  TORCH_CHECK(X.numel() > 0,
              "ternary_linear_packed_forward: X is empty (numel=0)");

  // Check W_packed is valid
  TORCH_CHECK(W_packed.dim() == 2,
              "ternary_linear_packed_forward: W_packed must be 2D [N, K_bytes], got shape ",
              format_shape(W_packed));
  TORCH_CHECK(W_packed.dtype() == torch::kUInt8,
              "ternary_linear_packed_forward: W_packed must be uint8, got ",
              W_packed.dtype());

  // Check scale is valid
  TORCH_CHECK(scale.dim() == 1,
              "ternary_linear_packed_forward: scale must be 1D [N], got shape ",
              format_shape(scale));

  // Get dimensions
  auto input_shape = X.sizes().vec();
  int K = input_shape.back();
  int B = X.numel() / K;
  int N = W_packed.size(0);
  int K_bytes = W_packed.size(1);

  // Validate in_features matches input
  TORCH_CHECK(K == in_features,
              "ternary_linear_packed_forward: input features K=", K,
              " doesn't match in_features=", in_features);

  // Validate packed size
  int expected_K_bytes = (in_features + 3) / 4;
  TORCH_CHECK(K_bytes == expected_K_bytes,
              "ternary_linear_packed_forward: W_packed K_bytes=", K_bytes,
              " doesn't match expected ceil(in_features/4)=", expected_K_bytes);

  TORCH_CHECK(scale.size(0) == N,
              "ternary_linear_packed_forward: scale size ", scale.size(0),
              " doesn't match N=", N,
              " (output features from W_packed)");

  // Check bias if provided
  if (bias.has_value()) {
    TORCH_CHECK(bias->dim() == 1,
                "ternary_linear_packed_forward: bias must be 1D [N], got shape ",
                format_shape(*bias));
    TORCH_CHECK(bias->size(0) == N,
                "ternary_linear_packed_forward: bias size ", bias->size(0),
                " doesn't match N=", N);
  }

  // ========================================
  // Device checks
  // ========================================

  TORCH_CHECK(X.is_cuda(),
              "ternary_linear_packed_forward: X must be on CUDA device, got ",
              X.device());
  TORCH_CHECK(W_packed.is_cuda(),
              "ternary_linear_packed_forward: W_packed must be on CUDA device, got ",
              W_packed.device());
  TORCH_CHECK(scale.is_cuda(),
              "ternary_linear_packed_forward: scale must be on CUDA device, got ",
              scale.device());

  // Check all tensors on same device
  TORCH_CHECK(X.device() == W_packed.device(),
              "ternary_linear_packed_forward: X and W_packed must be on same device. "
              "X on ", X.device(), ", W_packed on ", W_packed.device());
  TORCH_CHECK(X.device() == scale.device(),
              "ternary_linear_packed_forward: X and scale must be on same device. "
              "X on ", X.device(), ", scale on ", scale.device());

  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda(),
                "ternary_linear_packed_forward: bias must be on CUDA device, got ",
                bias->device());
    TORCH_CHECK(X.device() == bias->device(),
                "ternary_linear_packed_forward: X and bias must be on same device. "
                "X on ", X.device(), ", bias on ", bias->device());
  }

  // ========================================
  // Contiguity - make tensors contiguous
  // ========================================

  torch::Tensor X_contig = X.contiguous();
  torch::Tensor W_contig = W_packed.contiguous();
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

  torch::Tensor Y_2d = ternary_gemm_packed_cuda(X_2d, W_contig, scale_contig, bias_contig);

  // Reshape output back to [*, N]
  std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end() - 1);
  output_shape.push_back(N);

  return Y_2d.reshape(output_shape);
}

}  // namespace bittorch
