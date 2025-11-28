/**
 * Ternary GEMM kernel for BitTorch.
 *
 * Computes Y = X @ (W_tern * scale).T + bias
 * where W_tern contains ternary values {-1, 0, +1}.
 *
 * This is a baseline implementation (one thread per output element).
 * Future versions will add tiling, shared memory, and vectorization.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

#include "utils.cuh"

namespace bittorch {

/**
 * Baseline ternary GEMM kernel.
 *
 * Each thread computes one output element Y[b, n].
 *
 * @param X Input tensor [B, K]
 * @param W_tern Ternary weights [N, K] (values in {-1, 0, +1} stored as int8)
 * @param scale Per-channel scale [N]
 * @param bias Optional bias [N] (nullptr if not used)
 * @param Y Output tensor [B, N]
 * @param B Batch size
 * @param N Output features
 * @param K Input features
 */
template <typename scalar_t>
__global__ void ternary_gemm_kernel(
    const scalar_t* __restrict__ X,
    const int8_t* __restrict__ W_tern,
    const scalar_t* __restrict__ scale,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ Y,
    int B,
    int N,
    int K) {

  // Global thread indices
  int b = blockIdx.x * blockDim.x + threadIdx.x;  // batch index
  int n = blockIdx.y * blockDim.y + threadIdx.y;  // output feature index

  if (b >= B || n >= N) return;

  // Accumulate dot product in higher precision
  float acc = 0.0f;

  for (int k = 0; k < K; k++) {
    float x_val = static_cast<float>(X[b * K + k]);
    int8_t w_val = W_tern[n * K + k];  // -1, 0, or +1
    acc += x_val * static_cast<float>(w_val);
  }

  // Apply scale
  float s = static_cast<float>(scale[n]);
  acc *= s;

  // Apply bias if present
  if (bias != nullptr) {
    acc += static_cast<float>(bias[n]);
  }

  // Write output
  Y[b * N + n] = static_cast<scalar_t>(acc);
}

/**
 * Launch ternary GEMM kernel.
 *
 * @param X Input tensor [B, K]
 * @param W_tern Ternary weights [N, K] as int8
 * @param scale Per-channel scale [N]
 * @param bias Optional bias [N]
 * @return Output tensor [B, N]
 */
torch::Tensor ternary_gemm_cuda(
    torch::Tensor X,
    torch::Tensor W_tern,
    torch::Tensor scale,
    c10::optional<torch::Tensor> bias) {

  // Input validation
  TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
  TORCH_CHECK(W_tern.is_cuda(), "W_tern must be a CUDA tensor");
  TORCH_CHECK(scale.is_cuda(), "scale must be a CUDA tensor");
  TORCH_CHECK(X.dim() == 2, "X must be 2D [B, K]");
  TORCH_CHECK(W_tern.dim() == 2, "W_tern must be 2D [N, K]");
  TORCH_CHECK(scale.dim() == 1, "scale must be 1D [N]");

  int B = X.size(0);
  int K = X.size(1);
  int N = W_tern.size(0);

  TORCH_CHECK(W_tern.size(1) == K, "W_tern dim 1 must match X dim 1");
  TORCH_CHECK(scale.size(0) == N, "scale size must match W_tern dim 0");
  TORCH_CHECK(W_tern.scalar_type() == torch::kInt8, "W_tern must be int8");

  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias->dim() == 1, "bias must be 1D [N]");
    TORCH_CHECK(bias->size(0) == N, "bias size must match N");
  }

  // Create output tensor
  auto options = torch::TensorOptions()
      .dtype(X.dtype())
      .device(X.device());
  torch::Tensor Y = torch::empty({B, N}, options);

  // Launch kernel
  dim3 threads(16, 16);
  dim3 blocks(ceildiv(B, 16), ceildiv(N, 16));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(X.scalar_type(), "ternary_gemm_cuda", ([&] {
    ternary_gemm_kernel<scalar_t><<<blocks, threads>>>(
        X.data_ptr<scalar_t>(),
        W_tern.data_ptr<int8_t>(),
        scale.data_ptr<scalar_t>(),
        bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
        Y.data_ptr<scalar_t>(),
        B, N, K);
  }));

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "CUDA kernel launch failed: ", cudaGetErrorString(err));

  return Y;
}

}  // namespace bittorch
