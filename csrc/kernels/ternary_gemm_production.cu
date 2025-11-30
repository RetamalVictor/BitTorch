/**
 * Production Ternary GEMM Kernels for BitTorch.
 *
 * This file contains the two blessed production kernels:
 * - kernel_small_batch: Optimized for B <= 32 (inference)
 * - kernel_large_batch: Optimized for B > 32 (training)
 *
 * Both use TRANSPOSED weight layout: W_T [K_bytes, N]
 * This enables coalesced memory access for weight loading.
 *
 * Kernel selection:
 * - Automatic based on batch size (B <= 32 -> small, B > 32 -> large)
 * - Override via BITTORCH_KERNEL env var: "small", "large", or "auto"
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>

namespace bittorch {
namespace production {

//=============================================================================
// UTILITIES
//=============================================================================

template <typename T>
__host__ __device__ inline T ceildiv(T a, T b) {
    return (a + b - 1) / b;
}

/**
 * Branchless arithmetic decode: (code == 1) - (code == 2)
 * Compiles to predicated subtraction, no branches or memory access.
 */
__device__ __forceinline__ int decode_ternary(int code) {
    return (code == 1) - (code == 2);
}

//=============================================================================
// SMALL BATCH KERNEL (B <= 32, optimized for inference)
// Based on V2.3: TILE_K=128, transposed layout, arithmetic decode
//=============================================================================

namespace small_batch {

constexpr int TILE_B = 32;
constexpr int TILE_N = 32;
constexpr int TILE_K = 128;  // Large K tile to amortize sync overhead
constexpr int TILE_K_BYTES = TILE_K / 4;

template <typename scalar_t>
__global__ void kernel(
    const scalar_t* __restrict__ X,
    const uint8_t* __restrict__ W_T,   // [K_bytes, N] transposed
    const scalar_t* __restrict__ scale,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ Y,
    int B, int N, int K, int K_bytes) {

    __shared__ float X_tile[TILE_B][TILE_K + 1];   // +1 to avoid bank conflicts
    __shared__ float W_tile[TILE_K][TILE_N + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int global_b = blockIdx.x * TILE_B + ty;
    const int global_n = blockIdx.y * TILE_N + tx;

    // Pre-load scale and bias for this output channel
    const float s = (global_n < N) ? static_cast<float>(scale[global_n]) : 0.0f;
    const float b_val = (bias != nullptr && global_n < N) ? static_cast<float>(bias[global_n]) : 0.0f;

    float acc = 0.0f;

    const int num_k_tiles = ceildiv(K, TILE_K);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k_offset = kt * TILE_K;

        // Load X tile - each thread loads 4 values (TILE_K/32 = 4)
        #pragma unroll 4
        for (int i = 0; i < 4; i++) {
            const int load_k = k_offset + tx + i * 32;
            if (global_b < B && load_k < K) {
                X_tile[ty][tx + i * 32] = static_cast<float>(X[global_b * K + load_k]);
            } else {
                X_tile[ty][tx + i * 32] = 0.0f;
            }
        }

        // Load W tile - cooperative loading with arithmetic decode
        // TILE_K_BYTES * TILE_N = 32 * 32 = 1024 bytes total
        {
            const int linear_id = ty * TILE_N + tx;

            if (linear_id < TILE_K_BYTES * TILE_N) {
                const int byte_row = linear_id / TILE_N;
                const int n_col = linear_id % TILE_N;

                const int k_byte_global = (k_offset / 4) + byte_row;
                const int global_n_load = blockIdx.y * TILE_N + n_col;

                if (k_byte_global < K_bytes && global_n_load < N) {
                    // TRANSPOSED access: W_T[k_byte, n]
                    const uint8_t packed = W_T[k_byte_global * N + global_n_load];
                    const int k_base = byte_row * 4;

                    // Arithmetic decode - branchless
                    W_tile[k_base + 0][n_col] = static_cast<float>(decode_ternary((packed >> 0) & 0x03));
                    W_tile[k_base + 1][n_col] = static_cast<float>(decode_ternary((packed >> 2) & 0x03));
                    W_tile[k_base + 2][n_col] = static_cast<float>(decode_ternary((packed >> 4) & 0x03));
                    W_tile[k_base + 3][n_col] = static_cast<float>(decode_ternary((packed >> 6) & 0x03));
                } else {
                    const int k_base = byte_row * 4;
                    W_tile[k_base + 0][n_col] = 0.0f;
                    W_tile[k_base + 1][n_col] = 0.0f;
                    W_tile[k_base + 2][n_col] = 0.0f;
                    W_tile[k_base + 3][n_col] = 0.0f;
                }
            }
        }

        __syncthreads();

        // Compute - larger unroll for TILE_K=128
        if (global_b < B && global_n < N) {
            const int k_limit = (TILE_K < (K - k_offset)) ? TILE_K : (K - k_offset);
            #pragma unroll 16
            for (int k = 0; k < k_limit; k++) {
                acc += X_tile[ty][k] * W_tile[k][tx];
            }
        }

        __syncthreads();
    }

    // Write output with scale and bias
    if (global_b < B && global_n < N) {
        Y[global_b * N + global_n] = static_cast<scalar_t>(acc * s + b_val);
    }
}

}  // namespace small_batch

//=============================================================================
// LARGE BATCH KERNEL (B > 32, optimized for training)
// Based on V1 tiled, but ported to transposed layout
//=============================================================================

namespace large_batch {

constexpr int TILE_B = 32;
constexpr int TILE_N = 32;
constexpr int TILE_K = 32;  // Smaller K tile, more parallelism
constexpr int TILE_K_BYTES = TILE_K / 4;

template <typename scalar_t>
__global__ void kernel(
    const scalar_t* __restrict__ X,
    const uint8_t* __restrict__ W_T,   // [K_bytes, N] transposed
    const scalar_t* __restrict__ scale,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ Y,
    int B, int N, int K, int K_bytes) {

    __shared__ float X_tile[TILE_B][TILE_K + 1];
    __shared__ float W_tile[TILE_K][TILE_N + 1];  // float for faster compute

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int global_b = blockIdx.x * TILE_B + ty;
    const int global_n = blockIdx.y * TILE_N + tx;

    float acc = 0.0f;

    const int num_k_tiles = ceildiv(K, TILE_K);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k_offset = kt * TILE_K;

        // Load X tile
        {
            const int load_k = k_offset + tx;
            if (global_b < B && load_k < K) {
                X_tile[ty][tx] = static_cast<float>(X[global_b * K + load_k]);
            } else {
                X_tile[ty][tx] = 0.0f;
            }
        }

        // Load W tile from TRANSPOSED layout
        // tx maps to N (output channel), ty maps to K bytes
        // Use linear indexing: each thread loads one byte, unpacks 4 weights
        {
            const int linear_id = ty * TILE_N + tx;  // 0..1023

            if (linear_id < TILE_K_BYTES * TILE_N) {
                const int byte_row = linear_id / TILE_N;  // which K byte (0..7)
                const int n_col = linear_id % TILE_N;     // which N column (0..31)

                const int k_byte_global = (k_offset / 4) + byte_row;
                const int global_n_load = blockIdx.y * TILE_N + n_col;

                if (k_byte_global < K_bytes && global_n_load < N) {
                    // TRANSPOSED access: W_T[k_byte, n]
                    const uint8_t packed = W_T[k_byte_global * N + global_n_load];
                    const int k_base = byte_row * 4;

                    // Decode to float
                    W_tile[k_base + 0][n_col] = static_cast<float>(decode_ternary((packed >> 0) & 0x03));
                    W_tile[k_base + 1][n_col] = static_cast<float>(decode_ternary((packed >> 2) & 0x03));
                    W_tile[k_base + 2][n_col] = static_cast<float>(decode_ternary((packed >> 4) & 0x03));
                    W_tile[k_base + 3][n_col] = static_cast<float>(decode_ternary((packed >> 6) & 0x03));
                } else {
                    const int k_base = byte_row * 4;
                    W_tile[k_base + 0][n_col] = 0.0f;
                    W_tile[k_base + 1][n_col] = 0.0f;
                    W_tile[k_base + 2][n_col] = 0.0f;
                    W_tile[k_base + 3][n_col] = 0.0f;
                }
            }
        }

        __syncthreads();

        // Compute
        if (global_b < B && global_n < N) {
            const int k_limit = (TILE_K < (K - k_offset)) ? TILE_K : (K - k_offset);
            #pragma unroll 8
            for (int k = 0; k < k_limit; k++) {
                acc += X_tile[ty][k] * W_tile[k][tx];
            }
        }

        __syncthreads();
    }

    // Write output
    if (global_b < B && global_n < N) {
        const float s = static_cast<float>(scale[global_n]);
        float result = acc * s;

        if (bias != nullptr) {
            result += static_cast<float>(bias[global_n]);
        }

        Y[global_b * N + global_n] = static_cast<scalar_t>(result);
    }
}

}  // namespace large_batch

//=============================================================================
// DISPATCHER
//=============================================================================

/**
 * Determine which kernel to use based on batch size and env override.
 * Returns true for small_batch kernel, false for large_batch kernel.
 */
bool use_small_batch_kernel(int B) {
    const char* env = std::getenv("BITTORCH_KERNEL");

    if (env != nullptr) {
        if (std::strcmp(env, "small") == 0) return true;
        if (std::strcmp(env, "large") == 0) return false;
        // "auto" or unknown -> fall through to heuristic
    }

    // Heuristic: B <= 32 uses small-batch kernel
    return B <= 32;
}

/**
 * Production ternary GEMM launcher.
 *
 * Computes Y = X @ (W_T * scale).T + bias
 *
 * @param X Input tensor [B, K]
 * @param W_T Packed ternary weights [K_bytes, N] in TRANSPOSED layout
 * @param scale Per-channel scale [N]
 * @param bias Optional bias [N]
 * @return Output tensor [B, N]
 */
torch::Tensor ternary_gemm_production_cuda(
    torch::Tensor X,
    torch::Tensor W_T,
    torch::Tensor scale,
    c10::optional<torch::Tensor> bias) {

    // Input validation
    TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(W_T.is_cuda(), "W_T must be a CUDA tensor");
    TORCH_CHECK(scale.is_cuda(), "scale must be a CUDA tensor");
    TORCH_CHECK(X.dim() == 2, "X must be 2D [B, K]");
    TORCH_CHECK(W_T.dim() == 2, "W_T must be 2D [K_bytes, N]");
    TORCH_CHECK(scale.dim() == 1, "scale must be 1D [N]");
    TORCH_CHECK(W_T.scalar_type() == torch::kUInt8, "W_T must be uint8");

    const int B = X.size(0);
    const int K = X.size(1);
    const int K_bytes = W_T.size(0);
    const int N = W_T.size(1);

    // Validate packed size
    const int expected_K_bytes = (K + 3) / 4;
    TORCH_CHECK(K_bytes == expected_K_bytes,
                "W_T dim 0 (", K_bytes, ") doesn't match expected ceil(K/4) = ",
                expected_K_bytes, " for K=", K);

    TORCH_CHECK(scale.size(0) == N, "scale size must match W_T dim 1 (N)");

    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->dim() == 1, "bias must be 1D [N]");
        TORCH_CHECK(bias->size(0) == N, "bias size must match N");
    }

    // Create output tensor
    auto Y = torch::empty({B, N}, X.options());

    // Select kernel
    const bool use_small = use_small_batch_kernel(B);

    if (use_small) {
        // Small batch kernel: TILE_K=128, optimized for B <= 32
        dim3 threads(small_batch::TILE_N, small_batch::TILE_B);
        dim3 blocks(ceildiv(B, small_batch::TILE_B), ceildiv(N, small_batch::TILE_N));

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(X.scalar_type(), "ternary_gemm_small_batch", ([&] {
            small_batch::kernel<scalar_t><<<blocks, threads>>>(
                X.data_ptr<scalar_t>(),
                W_T.data_ptr<uint8_t>(),
                scale.data_ptr<scalar_t>(),
                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                Y.data_ptr<scalar_t>(),
                B, N, K, K_bytes);
        }));
    } else {
        // Large batch kernel: TILE_K=32, optimized for B > 32
        dim3 threads(large_batch::TILE_N, large_batch::TILE_B);
        dim3 blocks(ceildiv(B, large_batch::TILE_B), ceildiv(N, large_batch::TILE_N));

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(X.scalar_type(), "ternary_gemm_large_batch", ([&] {
            large_batch::kernel<scalar_t><<<blocks, threads>>>(
                X.data_ptr<scalar_t>(),
                W_T.data_ptr<uint8_t>(),
                scale.data_ptr<scalar_t>(),
                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
                Y.data_ptr<scalar_t>(),
                B, N, K, K_bytes);
        }));
    }

    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return Y;
}

}  // namespace production
}  // namespace bittorch
