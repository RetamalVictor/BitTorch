/**
 * Ternary GEMM kernels for BitTorch.
 *
 * Computes Y = X @ (W_tern * scale).T + bias
 * where W_tern contains ternary values {-1, 0, +1}.
 *
 * Includes:
 * - Baseline kernel: One thread per output (for reference/debugging)
 * - Tiled kernel: Uses shared memory for better memory access patterns
 *
 * Kernel selection via BITTORCH_KERNEL env var: "baseline" or "tiled"
 * Default: tiled for shapes where it helps, baseline for tiny shapes.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdlib>

namespace bittorch {

// Inline utility - ceiling division
template <typename T>
__host__ __device__ inline T ceildiv(T a, T b) {
  return (a + b - 1) / b;
}

// Tile sizes for the optimized kernel
constexpr int TILE_M = 32;  // Tile size for batch dimension
constexpr int TILE_N = 32;  // Tile size for output features
constexpr int TILE_K = 32;  // Tile size for reduction dimension

//=============================================================================
// BASELINE KERNEL (one thread per output)
//=============================================================================

/**
 * Baseline ternary GEMM kernel.
 * Each thread computes one output element Y[b, n].
 * Simple but inefficient - each K-element is loaded from global memory.
 */
template <typename scalar_t>
__global__ void ternary_gemm_baseline_kernel(
    const scalar_t* __restrict__ X,
    const int8_t* __restrict__ W_tern,
    const scalar_t* __restrict__ scale,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ Y,
    int B,
    int N,
    int K) {

  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;

  if (b >= B || n >= N) return;

  float acc = 0.0f;

  for (int k = 0; k < K; k++) {
    float x_val = static_cast<float>(X[b * K + k]);
    int8_t w_val = W_tern[n * K + k];
    acc += x_val * static_cast<float>(w_val);
  }

  float s = static_cast<float>(scale[n]);
  acc *= s;

  if (bias != nullptr) {
    acc += static_cast<float>(bias[n]);
  }

  Y[b * N + n] = static_cast<scalar_t>(acc);
}

//=============================================================================
// TILED KERNEL (shared memory optimization)
//=============================================================================

/**
 * Tiled ternary GEMM kernel with shared memory.
 *
 * Each thread block computes a TILE_M x TILE_N tile of the output.
 * We tile over K in chunks of TILE_K, loading X and W_tern tiles
 * into shared memory for efficient reuse.
 *
 * Memory access pattern:
 * - X tile [TILE_M, TILE_K]: Each row (batch sample) is reused TILE_N times
 * - W tile [TILE_N, TILE_K]: Each row (output channel) is reused TILE_M times
 *
 * Thread block: (TILE_N, TILE_M) threads
 * Each thread computes one output element within the tile.
 */
template <typename scalar_t>
__global__ void ternary_gemm_tiled_kernel(
    const scalar_t* __restrict__ X,
    const int8_t* __restrict__ W_tern,
    const scalar_t* __restrict__ scale,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ Y,
    int B,
    int N,
    int K) {

  // Shared memory for tiles
  __shared__ float X_tile[TILE_M][TILE_K];
  __shared__ int8_t W_tile[TILE_N][TILE_K];

  // Global output position
  int b = blockIdx.x * TILE_M + threadIdx.y;  // batch index
  int n = blockIdx.y * TILE_N + threadIdx.x;  // output feature index

  // Local thread position within block
  int ty = threadIdx.y;  // 0..TILE_M-1
  int tx = threadIdx.x;  // 0..TILE_N-1

  float acc = 0.0f;

  // Number of tiles along K dimension
  int num_k_tiles = ceildiv(K, TILE_K);

  for (int t = 0; t < num_k_tiles; t++) {
    int k_offset = t * TILE_K;

    // Cooperatively load X tile [TILE_M, TILE_K]
    // Each thread loads one element from its row
    {
      int load_b = blockIdx.x * TILE_M + ty;
      int load_k = k_offset + tx;

      if (load_b < B && load_k < K && tx < TILE_K) {
        X_tile[ty][tx] = static_cast<float>(X[load_b * K + load_k]);
      } else if (tx < TILE_K) {
        X_tile[ty][tx] = 0.0f;
      }
    }

    // Cooperatively load W tile [TILE_N, TILE_K]
    // Note: W is [N, K], we load rows of W into W_tile
    {
      int load_n = blockIdx.y * TILE_N + ty;
      int load_k = k_offset + tx;

      if (load_n < N && load_k < K && tx < TILE_K) {
        W_tile[ty][tx] = W_tern[load_n * K + load_k];
      } else if (tx < TILE_K) {
        W_tile[ty][tx] = 0;
      }
    }

    __syncthreads();

    // Compute partial dot product for this tile
    if (b < B && n < N) {
      int k_end = (TILE_K < (K - k_offset)) ? TILE_K : (K - k_offset);
      #pragma unroll 8
      for (int k = 0; k < k_end; k++) {
        // X_tile[ty][k] = X[b, k_offset+k]
        // W_tile[tx][k] = W[n, k_offset+k]
        acc += X_tile[ty][k] * static_cast<float>(W_tile[tx][k]);
      }
    }

    __syncthreads();
  }

  // Write output
  if (b < B && n < N) {
    float s = static_cast<float>(scale[n]);
    acc *= s;

    if (bias != nullptr) {
      acc += static_cast<float>(bias[n]);
    }

    Y[b * N + n] = static_cast<scalar_t>(acc);
  }
}

//=============================================================================
// KERNEL DISPATCH
//=============================================================================

/**
 * Check if we should use the baseline kernel.
 * Use baseline for very small shapes or when BITTORCH_KERNEL=baseline is set.
 */
bool use_baseline_kernel(int B, int N, int K) {
  // Check environment variable
  const char* env = std::getenv("BITTORCH_KERNEL");
  if (env != nullptr) {
    if (std::string(env) == "baseline") return true;
    if (std::string(env) == "tiled") return false;
  }

  // Use baseline for very small shapes where tiling overhead isn't worth it
  if (B * N < 256 || K < 32) {
    return true;
  }

  return false;  // Default to tiled kernel
}

/**
 * Launch ternary GEMM kernel.
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

  // Choose kernel
  bool use_baseline = use_baseline_kernel(B, N, K);

  if (use_baseline) {
    // Baseline kernel: 16x16 threads, one per output
    dim3 threads(16, 16);
    dim3 blocks(ceildiv(B, 16), ceildiv(N, 16));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(X.scalar_type(), "ternary_gemm_baseline", ([&] {
      ternary_gemm_baseline_kernel<scalar_t><<<blocks, threads>>>(
          X.data_ptr<scalar_t>(),
          W_tern.data_ptr<int8_t>(),
          scale.data_ptr<scalar_t>(),
          bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
          Y.data_ptr<scalar_t>(),
          B, N, K);
    }));
  } else {
    // Tiled kernel: TILE_N x TILE_M threads per block
    dim3 threads(TILE_N, TILE_M);
    dim3 blocks(ceildiv(B, TILE_M), ceildiv(N, TILE_N));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(X.scalar_type(), "ternary_gemm_tiled", ([&] {
      ternary_gemm_tiled_kernel<scalar_t><<<blocks, threads>>>(
          X.data_ptr<scalar_t>(),
          W_tern.data_ptr<int8_t>(),
          scale.data_ptr<scalar_t>(),
          bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
          Y.data_ptr<scalar_t>(),
          B, N, K);
    }));
  }

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "CUDA kernel launch failed: ", cudaGetErrorString(err));

  return Y;
}

}  // namespace bittorch
