#include <torch/extension.h>

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

  // Handle batched input with multiple leading dimensions
  // Reshape [*, K] -> [B, K] where B = product of leading dims
  auto input_shape = X.sizes().vec();
  int K = input_shape.back();
  int B = X.numel() / K;

  torch::Tensor X_2d = X.reshape({B, K});

  // Dispatch to CUDA
  TORCH_CHECK(X_2d.is_cuda(),
              "ternary_linear_forward currently requires CUDA tensors");

  torch::Tensor Y_2d = ternary_gemm_cuda(X_2d, W_tern, scale, bias);

  // Reshape output back to [*, N]
  int N = Y_2d.size(1);
  std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end() - 1);
  output_shape.push_back(N);

  return Y_2d.reshape(output_shape);
}

}  // namespace bittorch
