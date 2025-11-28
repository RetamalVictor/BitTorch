/**
 * Packed Ternary GEMM kernels for BitTorch.
 *
 * Computes Y = X @ (W_packed * scale).T + bias
 * where W_packed contains 2-bit ternary values packed into uint8:
 *   - 00 = 0
 *   - 01 = +1
 *   - 10 = -1
 *   - 11 = reserved
 *
 * This kernel reads packed weights directly without materializing
 * full float/int8 weight tensors, achieving ~16x memory reduction.
 *
 * Kernel selection via BITTORCH_PACKED_KERNEL env var: "baseline" or "tiled"
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdlib>

namespace bittorch {

// Inline utility - ceiling division
template <typename T>
__host__ __device__ inline T ceildiv_packed(T a, T b) {
  return (a + b - 1) / b;
}

// Tile sizes for the optimized kernel
constexpr int PACKED_TILE_M = 32;  // Tile size for batch dimension
constexpr int PACKED_TILE_N = 32;  // Tile size for output features
constexpr int PACKED_TILE_K = 32;  // Tile size for reduction dimension (unpacked)

/**
 * Decode a 2-bit ternary value to int.
 * Encoding: 00=0, 01=+1, 10=-1, 11=reserved (treated as 0)
 */
__device__ __forceinline__ int decode_ternary(int encoded) {
  // 0 -> 0, 1 -> +1, 2 -> -1, 3 -> 0 (reserved)
  return (encoded == 1) ? 1 : ((encoded == 2) ? -1 : 0);
}

/**
 * Unpack 4 ternary values from a single byte.
 * Layout: bits [1:0] = weight[0], [3:2] = weight[1], [5:4] = weight[2], [7:6] = weight[3]
 */
__device__ __forceinline__ void unpack_byte(uint8_t packed, int* out) {
  out[0] = decode_ternary((packed >> 0) & 0x03);
  out[1] = decode_ternary((packed >> 2) & 0x03);
  out[2] = decode_ternary((packed >> 4) & 0x03);
  out[3] = decode_ternary((packed >> 6) & 0x03);
}

//=============================================================================
// BASELINE KERNEL (one thread per output)
//=============================================================================

/**
 * Baseline packed ternary GEMM kernel.
 * Each thread computes one output element Y[b, n].
 * Unpacks weights on the fly from packed uint8 format.
 */
template <typename scalar_t>
__global__ void ternary_gemm_packed_baseline_kernel(
    const scalar_t* __restrict__ X,
    const uint8_t* __restrict__ W_packed,  // [N, K_bytes] packed
    const scalar_t* __restrict__ scale,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ Y,
    int B,
    int N,
    int K,           // Original in_features (unpacked)
    int K_bytes) {   // Packed bytes per row = ceil(K/4)

  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;

  if (b >= B || n >= N) return;

  float acc = 0.0f;

  // Process 4 weights at a time (one byte)
  int full_bytes = K / 4;
  int remainder = K % 4;

  for (int byte_idx = 0; byte_idx < full_bytes; byte_idx++) {
    uint8_t packed = W_packed[n * K_bytes + byte_idx];
    int w[4];
    unpack_byte(packed, w);

    int k_base = byte_idx * 4;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      float x_val = static_cast<float>(X[b * K + k_base + i]);
      // Use add/sub instead of multiply for ternary
      if (w[i] == 1) {
        acc += x_val;
      } else if (w[i] == -1) {
        acc -= x_val;
      }
      // w[i] == 0: skip (no-op)
    }
  }

  // Handle remainder (if K not divisible by 4)
  if (remainder > 0) {
    uint8_t packed = W_packed[n * K_bytes + full_bytes];
    int w[4];
    unpack_byte(packed, w);

    int k_base = full_bytes * 4;
    for (int i = 0; i < remainder; i++) {
      float x_val = static_cast<float>(X[b * K + k_base + i]);
      if (w[i] == 1) {
        acc += x_val;
      } else if (w[i] == -1) {
        acc -= x_val;
      }
    }
  }

  // Apply scale
  float s = static_cast<float>(scale[n]);
  acc *= s;

  // Apply bias
  if (bias != nullptr) {
    acc += static_cast<float>(bias[n]);
  }

  Y[b * N + n] = static_cast<scalar_t>(acc);
}

//=============================================================================
// TILED KERNEL (shared memory optimization)
//=============================================================================

/**
 * Tiled packed ternary GEMM kernel with shared memory.
 *
 * Each thread block computes a TILE_M x TILE_N tile of the output.
 * We tile over K in chunks, loading X tiles and unpacked W tiles
 * into shared memory.
 *
 * The packed weights are loaded and unpacked cooperatively into
 * shared memory as int8 (already decoded ternary values).
 */
template <typename scalar_t>
__global__ void ternary_gemm_packed_tiled_kernel(
    const scalar_t* __restrict__ X,
    const uint8_t* __restrict__ W_packed,
    const scalar_t* __restrict__ scale,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ Y,
    int B,
    int N,
    int K,
    int K_bytes) {

  // Shared memory for tiles
  __shared__ float X_tile[PACKED_TILE_M][PACKED_TILE_K];
  __shared__ int8_t W_tile[PACKED_TILE_N][PACKED_TILE_K];

  // Global output position
  int b = blockIdx.x * PACKED_TILE_M + threadIdx.y;  // batch index
  int n = blockIdx.y * PACKED_TILE_N + threadIdx.x;  // output feature index

  // Local thread position within block
  int ty = threadIdx.y;  // 0..TILE_M-1
  int tx = threadIdx.x;  // 0..TILE_N-1

  float acc = 0.0f;

  // Number of tiles along K dimension
  int num_k_tiles = ceildiv_packed(K, PACKED_TILE_K);

  for (int t = 0; t < num_k_tiles; t++) {
    int k_offset = t * PACKED_TILE_K;

    // Cooperatively load X tile [TILE_M, TILE_K]
    {
      int load_b = blockIdx.x * PACKED_TILE_M + ty;
      int load_k = k_offset + tx;

      if (load_b < B && load_k < K && tx < PACKED_TILE_K) {
        X_tile[ty][tx] = static_cast<float>(X[load_b * K + load_k]);
      } else if (tx < PACKED_TILE_K) {
        X_tile[ty][tx] = 0.0f;
      }
    }

    // Cooperatively load and unpack W tile [TILE_N, TILE_K]
    // Each thread handles unpacking for its row (tx = n index within tile)
    // We need to load multiple k values per thread
    {
      int load_n = blockIdx.y * PACKED_TILE_N + ty;  // Use ty for N dimension here

      // Each thread loads and unpacks weights for one output channel
      // We process TILE_K / 4 bytes per channel within this tile
      if (load_n < N) {
        int k_start = k_offset + tx * 4;  // Each thread handles 4 k values
        if (k_start < K && tx < (PACKED_TILE_K / 4)) {
          int byte_idx = k_start / 4;
          uint8_t packed = W_packed[load_n * K_bytes + byte_idx];
          int w[4];
          unpack_byte(packed, w);

          // Store unpacked values
          int k_base = tx * 4;
          #pragma unroll
          for (int i = 0; i < 4; i++) {
            if (k_base + i < PACKED_TILE_K) {
              W_tile[ty][k_base + i] = static_cast<int8_t>(w[i]);
            }
          }
        } else if (tx < (PACKED_TILE_K / 4)) {
          // Zero padding
          int k_base = tx * 4;
          #pragma unroll
          for (int i = 0; i < 4; i++) {
            if (k_base + i < PACKED_TILE_K) {
              W_tile[ty][k_base + i] = 0;
            }
          }
        }
      } else if (ty < PACKED_TILE_N) {
        // Zero padding for invalid N
        int k_base = tx * 4;
        if (tx < (PACKED_TILE_K / 4)) {
          #pragma unroll
          for (int i = 0; i < 4; i++) {
            if (k_base + i < PACKED_TILE_K) {
              W_tile[ty][k_base + i] = 0;
            }
          }
        }
      }
    }

    __syncthreads();

    // Compute partial dot product for this tile
    if (b < B && n < N) {
      int k_end = (PACKED_TILE_K < (K - k_offset)) ? PACKED_TILE_K : (K - k_offset);
      #pragma unroll 8
      for (int k = 0; k < k_end; k++) {
        float x_val = X_tile[ty][k];
        int8_t w_val = W_tile[tx][k];
        // Use add/sub for ternary (though this micro-optimization
        // may not help much since the compiler may optimize anyway)
        if (w_val == 1) {
          acc += x_val;
        } else if (w_val == -1) {
          acc -= x_val;
        }
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
 * Check if we should use the baseline kernel for packed weights.
 */
bool use_packed_baseline_kernel(int B, int N, int K) {
  // Check environment variable
  const char* env = std::getenv("BITTORCH_PACKED_KERNEL");
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
 * Launch packed ternary GEMM kernel.
 *
 * @param X Input tensor [B, K], fp16/fp32
 * @param W_packed Packed ternary weights [N, K_bytes], uint8
 * @param scale Per-channel scale [N], fp16/fp32
 * @param bias Optional bias [N], fp16/fp32
 * @return Output tensor [B, N]
 */
torch::Tensor ternary_gemm_packed_cuda(
    torch::Tensor X,
    torch::Tensor W_packed,
    torch::Tensor scale,
    c10::optional<torch::Tensor> bias) {

  // Input validation
  TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
  TORCH_CHECK(W_packed.is_cuda(), "W_packed must be a CUDA tensor");
  TORCH_CHECK(scale.is_cuda(), "scale must be a CUDA tensor");
  TORCH_CHECK(X.dim() == 2, "X must be 2D [B, K]");
  TORCH_CHECK(W_packed.dim() == 2, "W_packed must be 2D [N, K_bytes]");
  TORCH_CHECK(scale.dim() == 1, "scale must be 1D [N]");

  int B = X.size(0);
  int K = X.size(1);
  int N = W_packed.size(0);
  int K_bytes = W_packed.size(1);

  // Validate packed size: K_bytes should be ceil(K/4)
  int expected_K_bytes = (K + 3) / 4;
  TORCH_CHECK(K_bytes == expected_K_bytes,
              "W_packed dim 1 (", K_bytes, ") doesn't match expected ceil(K/4) = ",
              expected_K_bytes, " for K=", K);

  TORCH_CHECK(scale.size(0) == N, "scale size must match W_packed dim 0");
  TORCH_CHECK(W_packed.scalar_type() == torch::kUInt8, "W_packed must be uint8");

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
  bool use_baseline = use_packed_baseline_kernel(B, N, K);

  if (use_baseline) {
    // Baseline kernel: 16x16 threads, one per output
    dim3 threads(16, 16);
    dim3 blocks(ceildiv_packed(B, 16), ceildiv_packed(N, 16));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(X.scalar_type(), "ternary_gemm_packed_baseline", ([&] {
      ternary_gemm_packed_baseline_kernel<scalar_t><<<blocks, threads>>>(
          X.data_ptr<scalar_t>(),
          W_packed.data_ptr<uint8_t>(),
          scale.data_ptr<scalar_t>(),
          bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
          Y.data_ptr<scalar_t>(),
          B, N, K, K_bytes);
    }));
  } else {
    // Tiled kernel: TILE_N x TILE_M threads per block
    dim3 threads(PACKED_TILE_N, PACKED_TILE_M);
    dim3 blocks(ceildiv_packed(B, PACKED_TILE_M), ceildiv_packed(N, PACKED_TILE_N));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(X.scalar_type(), "ternary_gemm_packed_tiled", ([&] {
      ternary_gemm_packed_tiled_kernel<scalar_t><<<blocks, threads>>>(
          X.data_ptr<scalar_t>(),
          W_packed.data_ptr<uint8_t>(),
          scale.data_ptr<scalar_t>(),
          bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
          Y.data_ptr<scalar_t>(),
          B, N, K, K_bytes);
    }));
  }

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "CUDA kernel launch failed: ", cudaGetErrorString(err));

  return Y;
}

}  // namespace bittorch
