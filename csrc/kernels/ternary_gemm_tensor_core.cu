/**
 * Tensor Core Ternary GEMM Kernel for BitTorch
 *
 * Hybrid approach: Store 2-bit packed, unpack to INT8 for Tensor Cores.
 *
 * Key features:
 * - LUT-based 2-bit to INT8 unpacking in shared memory
 * - V1: Float compute (baseline, no Tensor Cores)
 * - V2: mma.sync.aligned.m16n8k32 for INT8 Tensor Core operations
 *       Note: m16n8k16 is for FP16/BF16 only, INT8 requires m8n8k16 or m16n8k32
 * - Per-row activation quantization with scale tracking
 *
 * Weight layout: W_T [K_bytes, N] (transposed, packed)
 * Encoding: 00 = 0, 01 = +127, 10 = -127
 *
 * Scaling math:
 *   y = (acc * s_x / 127) * scale + bias
 *   where s_x = max(|x_row|) / 127 per-row activation scale
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

namespace bittorch {
namespace tensor_core {

//=============================================================================
// CONSTANTS AND CONFIGURATION
//=============================================================================

// Tile sizes (conservative, tuned for SM86)
constexpr int BLOCK_M = 64;   // Batch tile
constexpr int BLOCK_N = 64;   // Output features tile
constexpr int BLOCK_K = 32;   // Reduction tile (8 packed bytes)
constexpr int BLOCK_K_BYTES = BLOCK_K / 4;  // 8 bytes per K tile

// Thread block configuration: 256 threads (8 warps)
constexpr int THREADS_X = 32;
constexpr int THREADS_Y = 8;
constexpr int NUM_WARPS = THREADS_Y;

// MMA dimensions for INT8 on Ampere: m16n8k32
// NOTE: m16n8k16 is for FP16/BF16, NOT INT8!
// INT8 valid shapes on sm_86: m8n8k16, m16n8k32
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 32;  // K=32 for INT8 mma.sync

// Padding to avoid bank conflicts
constexpr int SMEM_PAD = 16;

//=============================================================================
// LUT FOR 2-BIT TO INT8 UNPACKING
//=============================================================================

// 256-entry LUT: each entry is 4 INT8 values packed into int32
// Stored in constant memory for fast broadcast access
__constant__ int32_t UNPACK_LUT[256];

// Flag to track if LUT is initialized
static bool lut_initialized = false;

// Host-side LUT initialization
void init_unpack_lut() {
    if (lut_initialized) return;

    int32_t lut[256];
    for (int byte = 0; byte < 256; byte++) {
        int8_t vals[4];
        for (int j = 0; j < 4; j++) {
            int code = (byte >> (2 * j)) & 0x03;
            // Encoding: 00 = 0, 01 = +127, 10 = -127
            vals[j] = (code == 1) ? 127 : (code == 2) ? -127 : 0;
        }
        memcpy(&lut[byte], vals, 4);
    }
    cudaMemcpyToSymbol(UNPACK_LUT, lut, sizeof(lut));
    lut_initialized = true;
}

//=============================================================================
// HELPER FUNCTIONS
//=============================================================================

template <typename T>
__host__ __device__ inline T ceildiv(T a, T b) {
    return (a + b - 1) / b;
}

// Warp-level reduction for max
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

//=============================================================================
// SIMPLIFIED TENSOR CORE KERNEL (V1: No MMA, uses LUT unpack + regular compute)
// This is a stepping stone to validate the LUT unpack path
//=============================================================================

// Number of columns per thread: BLOCK_N / THREADS_X = 64 / 32 = 2
constexpr int COLS_PER_THREAD = BLOCK_N / THREADS_X;
// Number of rows per thread: BLOCK_M / THREADS_Y = 64 / 8 = 8
constexpr int ROWS_PER_THREAD = BLOCK_M / THREADS_Y;

template <typename scalar_t>
__global__ void kernel_v1_lut_unpack(
    const scalar_t* __restrict__ X,      // [B, K] activations
    const uint8_t* __restrict__ W_T,     // [K_bytes, N] packed weights (transposed)
    const scalar_t* __restrict__ scale,  // [N] per-channel scale
    const scalar_t* __restrict__ bias,   // [N] optional bias
    scalar_t* __restrict__ Y,            // [B, N] output
    int B, int N, int K, int K_bytes
) {
    // Shared memory for tiles
    __shared__ float X_smem[BLOCK_M][BLOCK_K + SMEM_PAD];
    __shared__ int8_t W_smem[BLOCK_K][BLOCK_N + SMEM_PAD];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * THREADS_X + tx;

    const int block_b = blockIdx.x * BLOCK_M;
    const int block_n = blockIdx.y * BLOCK_N;

    // Each thread handles ROWS_PER_THREAD rows × COLS_PER_THREAD columns
    // Thread (tx, ty) handles:
    //   rows: [ty * ROWS_PER_THREAD, ty * ROWS_PER_THREAD + ROWS_PER_THREAD)
    //   cols: [tx * COLS_PER_THREAD, tx * COLS_PER_THREAD + COLS_PER_THREAD)

    // Pre-load scale and bias for this thread's columns
    float s[COLS_PER_THREAD];
    float b_val[COLS_PER_THREAD];
    #pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; c++) {
        int global_n = block_n + tx * COLS_PER_THREAD + c;
        s[c] = (global_n < N) ? static_cast<float>(scale[global_n]) : 0.0f;
        b_val[c] = (bias != nullptr && global_n < N) ? static_cast<float>(bias[global_n]) : 0.0f;
    }

    // Accumulators: ROWS_PER_THREAD rows × COLS_PER_THREAD columns
    float acc[ROWS_PER_THREAD][COLS_PER_THREAD];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; r++) {
        #pragma unroll
        for (int c = 0; c < COLS_PER_THREAD; c++) {
            acc[r][c] = 0.0f;
        }
    }

    const int num_k_tiles = ceildiv(K, BLOCK_K);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k_offset = kt * BLOCK_K;

        // ===== Stage 1: Load X tile to shared memory =====
        // 256 threads load 64 rows × 32 K values = 2048 elements
        // Each thread loads 2048/256 = 8 elements
        {
            const int elems_per_thread = (BLOCK_M * BLOCK_K) / (THREADS_X * THREADS_Y);
            #pragma unroll
            for (int i = 0; i < elems_per_thread; i++) {
                int elem_idx = tid + i * (THREADS_X * THREADS_Y);
                int row = elem_idx / BLOCK_K;
                int col = elem_idx % BLOCK_K;
                int global_row = block_b + row;
                int global_col = k_offset + col;

                if (global_row < B && global_col < K) {
                    X_smem[row][col] = static_cast<float>(X[global_row * K + global_col]);
                } else {
                    X_smem[row][col] = 0.0f;
                }
            }
        }

        // ===== Stage 2: Load and unpack W tile =====
        // Load packed bytes, unpack via LUT
        {
            // 8 K_bytes × 64 N = 512 packed bytes
            const int bytes_per_tile = BLOCK_K_BYTES * BLOCK_N;
            const int bytes_per_thread = ceildiv(bytes_per_tile, THREADS_X * THREADS_Y);

            #pragma unroll
            for (int i = 0; i < bytes_per_thread; i++) {
                int byte_idx = tid + i * (THREADS_X * THREADS_Y);
                if (byte_idx >= bytes_per_tile) break;

                int kb = byte_idx / BLOCK_N;  // which K_byte row
                int n = byte_idx % BLOCK_N;   // which N column

                int global_kb = (k_offset / 4) + kb;
                int global_n_idx = block_n + n;

                uint8_t packed = 0;
                if (global_kb < K_bytes && global_n_idx < N) {
                    packed = W_T[global_kb * N + global_n_idx];
                }

                // LUT unpack: 1 byte -> 4 INT8 values
                int32_t expanded = UNPACK_LUT[packed];
                int8_t* vals = (int8_t*)&expanded;

                // Write 4 INT8 values to shared memory
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int k_local = kb * 4 + j;
                    if (k_local < BLOCK_K) {
                        W_smem[k_local][n] = vals[j];
                    }
                }
            }
        }

        __syncthreads();

        // ===== Stage 3: Compute =====
        // Each thread computes ROWS_PER_THREAD × COLS_PER_THREAD outputs
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; r++) {
            int row = ty * ROWS_PER_THREAD + r;
            #pragma unroll
            for (int k = 0; k < BLOCK_K; k++) {
                if (k_offset + k < K) {
                    float x_val = X_smem[row][k];
                    #pragma unroll
                    for (int c = 0; c < COLS_PER_THREAD; c++) {
                        int col = tx * COLS_PER_THREAD + c;
                        acc[r][c] += x_val * static_cast<float>(W_smem[k][col]);
                    }
                }
            }
        }

        __syncthreads();
    }

    // ===== Epilogue: Scale and store =====
    // W_smem contains INT8 values in [-127, 127] representing {-1, 0, +1}
    // So we divide by 127 to get back to ternary scale
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; r++) {
        int row = ty * ROWS_PER_THREAD + r;
        int global_row = block_b + row;

        #pragma unroll
        for (int c = 0; c < COLS_PER_THREAD; c++) {
            int col = tx * COLS_PER_THREAD + c;
            int global_n = block_n + col;

            if (global_row < B && global_n < N) {
                float result = (acc[r][c] / 127.0f) * s[c] + b_val[c];
                Y[global_row * N + global_n] = static_cast<scalar_t>(result);
            }
        }
    }
}

//=============================================================================
// TENSOR CORE KERNEL (V2: With mma.sync)
// Uses m16n8k16 INT8 Tensor Core instructions
//=============================================================================

// MMA helper: m16n8k32 INT8 -> INT32
// This is the correct shape for INT8 on Ampere (sm_86)!
// Fragment sizes for m16n8k32:
//   A: 4 int32 registers (16 int8 values each = 64 int8 total)
//   B: 2 int32 registers (8 int8 values each = 16 int8 total)
//   C/D: 4 int32 registers (4 int32 values)
__device__ __forceinline__ void mma_m16n8k32_s8(
    int32_t& d0, int32_t& d1, int32_t& d2, int32_t& d3,  // accumulator output
    int32_t a0, int32_t a1, int32_t a2, int32_t a3,       // A fragment
    int32_t b0, int32_t b1,                               // B fragment
    int32_t c0, int32_t c1, int32_t c2, int32_t c3        // accumulator input
) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, "     // D: 4 registers
        "{%4, %5, %6, %7}, "     // A: 4 registers
        "{%8, %9}, "             // B: 2 registers
        "{%10, %11, %12, %13};\n" // C: 4 registers
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(c0), "r"(c1), "r"(c2), "r"(c3)
    );
}

//=============================================================================
// TENSOR CORE KERNEL V2: With mma.sync
// Uses m16n8k32 INT8 Tensor Core instructions (correct shape for INT8!)
//
// Previously failed with m16n8k16 which is only valid for FP16/BF16.
// See Journal/kernel_optimization/018 for investigation details.
//=============================================================================

// Thread/Warp mapping for 64x64 output tile:
// - 8 warps (256 threads)
// - Each warp handles a 16x32 output region (4 mma ops in N direction)
// - Warp layout: 4 warps in M direction, 2 in N direction
// - MMA shape: m16n8k32 -> 16 rows x 8 cols x 32 reduction per instruction
//
// Fragment layout for m16n8k32 INT8:
// - A fragment (row-major): 4 int32s per thread
// - B fragment (col-major): 2 int32s per thread
// - C/D fragment: 4 int32s per thread

// V2 tile configuration - optimized for mma alignment
constexpr int V2_BLOCK_M = 64;    // Must be multiple of MMA_M (16)
constexpr int V2_BLOCK_N = 64;    // Must be multiple of MMA_N (8)
constexpr int V2_BLOCK_K = 32;    // Must be multiple of MMA_K (32) - exactly one mma per K tile

// Warp tile sizes (each warp computes this much)
constexpr int WARP_M = 16;        // One MMA in M
constexpr int WARP_N = 32;        // Four MMAs in N
constexpr int WARPS_M = V2_BLOCK_M / WARP_M;  // 4 warps in M
constexpr int WARPS_N = V2_BLOCK_N / WARP_N;  // 2 warps in N

// Shared memory padding for bank conflict avoidance
constexpr int V2_SMEM_PAD_A = 16;  // Padding for A tile (int8)
constexpr int V2_SMEM_PAD_B = 16;  // Padding for B tile (int8)

// Helper: Load 4 int8 values from shared memory into an int32 register
__device__ __forceinline__ int32_t load_int8x4(const int8_t* ptr) {
    return *reinterpret_cast<const int32_t*>(ptr);
}

// Helper: Quantize FP32 to INT8 with scale
__device__ __forceinline__ int8_t quantize_to_int8(float val, float inv_scale) {
    float scaled = val * inv_scale;
    scaled = fmaxf(-127.0f, fminf(127.0f, roundf(scaled)));
    return static_cast<int8_t>(scaled);
}

template <typename scalar_t>
__global__ void kernel_v2_tensor_core(
    const scalar_t* __restrict__ X,      // [B, K] activations
    const uint8_t* __restrict__ W_T,     // [K_bytes, N] packed weights (transposed)
    const scalar_t* __restrict__ scale,  // [N] per-channel scale
    const scalar_t* __restrict__ bias,   // [N] optional bias
    scalar_t* __restrict__ Y,            // [B, N] output
    int B, int N, int K, int K_bytes
) {
    // Shared memory layout:
    // - A_smem: [BLOCK_M][BLOCK_K + PAD] int8 (quantized activations)
    // - B_smem: [BLOCK_K][BLOCK_N + PAD] int8 (unpacked weights)
    // - row_scale: [BLOCK_M] float (per-row activation scales)
    __shared__ int8_t A_smem[V2_BLOCK_M][V2_BLOCK_K + V2_SMEM_PAD_A];
    __shared__ int8_t B_smem[V2_BLOCK_K][V2_BLOCK_N + V2_SMEM_PAD_B];
    __shared__ float row_scale[V2_BLOCK_M];

    // Also need temporary FP32 storage for activation loading
    __shared__ float X_fp32[V2_BLOCK_M][V2_BLOCK_K + SMEM_PAD];

    const int tx = threadIdx.x;  // Lane within warp (0-31)
    const int ty = threadIdx.y;  // Warp index (0-7)
    const int tid = ty * THREADS_X + tx;
    const int warp_id = ty;
    const int lane_id = tx;

    // Block position
    const int block_b = blockIdx.x * V2_BLOCK_M;
    const int block_n = blockIdx.y * V2_BLOCK_N;

    // Warp position within block
    const int warp_m = warp_id / WARPS_N;  // 0-3
    const int warp_n = warp_id % WARPS_N;  // 0-1

    // Initialize accumulators (4 mma outputs per warp in N direction, 1 in M)
    // Each mma gives 16x8 output, warp does 16x32, so 4 mma ops
    int32_t acc[4][4];  // [4 mma ops in N][4 values per mma]
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            acc[i][j] = 0;
        }
    }

    const int num_k_tiles = ceildiv(K, V2_BLOCK_K);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k_offset = kt * V2_BLOCK_K;

        // ===== Stage 1: Load activations as FP32 =====
        {
            const int elems_per_thread = (V2_BLOCK_M * V2_BLOCK_K) / (THREADS_X * THREADS_Y);
            #pragma unroll
            for (int i = 0; i < elems_per_thread; i++) {
                int elem_idx = tid + i * (THREADS_X * THREADS_Y);
                int row = elem_idx / V2_BLOCK_K;
                int col = elem_idx % V2_BLOCK_K;
                int global_row = block_b + row;
                int global_col = k_offset + col;

                if (global_row < B && global_col < K) {
                    X_fp32[row][col] = static_cast<float>(X[global_row * K + global_col]);
                } else {
                    X_fp32[row][col] = 0.0f;
                }
            }
        }

        // ===== Stage 2: Compute per-row max and quantize activations =====
        __syncthreads();

        // Each warp computes row max for some rows
        // With 8 warps and 64 rows, each warp handles 8 rows
        // But we need warp-level reduction over K dimension
        {
            // Each thread in a warp handles a different K element
            // Warp handles rows [warp_id * 8, warp_id * 8 + 8)
            for (int local_row = 0; local_row < 8; local_row++) {
                int row = warp_id * 8 + local_row;
                if (row < V2_BLOCK_M) {
                    // Each lane handles one element at a time, reduce over K
                    float max_val = 0.0f;
                    for (int k = lane_id; k < V2_BLOCK_K; k += 32) {
                        max_val = fmaxf(max_val, fabsf(X_fp32[row][k]));
                    }
                    max_val = warp_reduce_max(max_val);

                    // Lane 0 writes the scale
                    if (lane_id == 0) {
                        float s = (max_val > 1e-6f) ? (max_val / 127.0f) : 1.0f;
                        row_scale[row] = s;
                    }
                }
            }
        }
        __syncthreads();

        // Now quantize to INT8
        {
            const int elems_per_thread = (V2_BLOCK_M * V2_BLOCK_K) / (THREADS_X * THREADS_Y);
            #pragma unroll
            for (int i = 0; i < elems_per_thread; i++) {
                int elem_idx = tid + i * (THREADS_X * THREADS_Y);
                int row = elem_idx / V2_BLOCK_K;
                int col = elem_idx % V2_BLOCK_K;

                float inv_scale = 1.0f / row_scale[row];
                A_smem[row][col] = quantize_to_int8(X_fp32[row][col], inv_scale);
            }
        }

        // ===== Stage 3: Load and unpack weights =====
        {
            const int bytes_per_tile = (V2_BLOCK_K / 4) * V2_BLOCK_N;
            const int bytes_per_thread = ceildiv(bytes_per_tile, THREADS_X * THREADS_Y);

            #pragma unroll
            for (int i = 0; i < bytes_per_thread; i++) {
                int byte_idx = tid + i * (THREADS_X * THREADS_Y);
                if (byte_idx >= bytes_per_tile) break;

                int kb = byte_idx / V2_BLOCK_N;
                int n = byte_idx % V2_BLOCK_N;

                int global_kb = (k_offset / 4) + kb;
                int global_n_idx = block_n + n;

                uint8_t packed = 0;
                if (global_kb < K_bytes && global_n_idx < N) {
                    packed = W_T[global_kb * N + global_n_idx];
                }

                // LUT unpack
                int32_t expanded = UNPACK_LUT[packed];
                int8_t* vals = (int8_t*)&expanded;

                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int k_local = kb * 4 + j;
                    if (k_local < V2_BLOCK_K) {
                        B_smem[k_local][n] = vals[j];
                    }
                }
            }
        }

        __syncthreads();

        // ===== Stage 4: Tensor Core compute using mma.sync =====
        // Each warp does 1 mma in M, 4 in N
        // With MMA_K=32 and BLOCK_K=32, only one iteration needed per K tile

        // Warp's output position
        int warp_row_base = warp_m * WARP_M;       // Row offset: 0, 16, 32, 48
        int warp_col_base = warp_n * WARP_N;       // Col offset: 0 or 32

        // Iterate over K dimension in chunks of MMA_K (32)
        // With BLOCK_K=32, this loop runs exactly once
        for (int kk = 0; kk < V2_BLOCK_K; kk += MMA_K) {
            // Load A fragment for m16n8k32 (row-major):
            // A is 16 rows × 32 K = 512 int8 / 32 threads = 16 int8 per thread = 4 int32 regs
            //
            // Using 4-lane groups (same as D fragment):
            // groupNum = lane_id / 4 (0-7) → rows
            // groupIdx = lane_id % 4 (0-3) → K position
            //
            // HYPOTHESIS: K is chunked in 4s, and each groupIdx maps to 8 K values
            // spread across the 32 K total. Based on test showing only K=0-3 work.

            int groupNum = lane_id / 4;  // 0-7: row group (same as D output)
            int groupIdx = lane_id % 4;  // 0-3: K position

            int32_t frag_a[4];
            int a_row_lo = warp_row_base + groupNum;       // rows 0-7
            int a_row_hi = warp_row_base + groupNum + 8;   // rows 8-15

            // Try K mapping: each groupIdx handles K positions spread across 32
            // groupIdx=0: K=0,1,2,3, 16,17,18,19
            // groupIdx=1: K=4,5,6,7, 20,21,22,23
            // groupIdx=2: K=8,9,10,11, 24,25,26,27
            // groupIdx=3: K=12,13,14,15, 28,29,30,31
            int k_base_lo = kk + groupIdx * 4;             // K=0,4,8,12
            int k_base_hi = kk + groupIdx * 4 + 16;        // K=16,20,24,28

            frag_a[0] = load_int8x4(&A_smem[a_row_lo][k_base_lo]);      // row_lo, K_lo chunk
            frag_a[1] = load_int8x4(&A_smem[a_row_lo][k_base_hi]);      // row_lo, K_hi chunk
            frag_a[2] = load_int8x4(&A_smem[a_row_hi][k_base_lo]);      // row_hi, K_lo chunk
            frag_a[3] = load_int8x4(&A_smem[a_row_hi][k_base_hi]);      // row_hi, K_hi chunk

            // Do 4 MMA operations for 4 different N positions
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                int n_offset = warp_col_base + ni * MMA_N;

                // Load B fragment for m16n8k32 (col-major):
                // B is 32 K × 8 N = 256 int8 / 32 threads = 8 int8 per thread = 2 int32 regs
                //
                // Fragment layout MUST match output fragment layout!
                // Output columns: groupIdx * 2 and groupIdx * 2 + 1
                // So threads with same groupIdx need B data for same columns
                //
                // B fragment mapping:
                //   - groupNum (lane_id / 4): determines which K slice
                //   - groupIdx (lane_id % 4): determines which N column pair
                //   Each thread loads K values for ONE column
                //   - Threads 0,1,2,3 load for N cols (via their K slices)
                //   - But need to coordinate which column each thread loads
                //
                // Revised: Each thread group (same groupNum) handles all 8 columns
                //          groupIdx determines K slice, groupNum determines column
                //          Actually: swap the roles for B

                // For B fragment (col-major), matching A's K spread pattern:
                // - groupNum (lane_id / 4) determines N column: 0-7
                // - groupIdx (lane_id % 4) determines K slice
                //
                // Same K spread as A: each groupIdx handles K positions spread across 32
                // B needs 2 int32 regs = 8 int8 values per thread
                int b_n = n_offset + groupNum;  // B column 0-7
                int b_k_lo = kk + groupIdx * 4;      // K=0,4,8,12
                int b_k_hi = kk + groupIdx * 4 + 16; // K=16,20,24,28

                // Load B values - B_smem is [K][N], load K positions for this column
                int8_t b_vals0[4], b_vals1[4];
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    b_vals0[i] = B_smem[b_k_lo + i][b_n];   // K_lo chunk
                    b_vals1[i] = B_smem[b_k_hi + i][b_n];   // K_hi chunk
                }
                int32_t frag_b0, frag_b1;
                memcpy(&frag_b0, b_vals0, 4);
                memcpy(&frag_b1, b_vals1, 4);

                // Execute MMA with correct m16n8k32 instruction
                mma_m16n8k32_s8(
                    acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3],
                    frag_a[0], frag_a[1], frag_a[2], frag_a[3],
                    frag_b0, frag_b1,
                    acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3]
                );
            }
        }

        __syncthreads();
    }

    // ===== Epilogue: Convert INT32 accumulators to FP32 output =====
    // acc = sum of (x_int8 * w_int8) = sum of ((x/s_x) * (w_tern*127))
    // To get y = x @ w_tern.T * scale:
    //   y = acc * s_x / 127 * scale

    // For m16n8k32 output, the D fragment layout is:
    // D is 16 rows × 8 cols
    //
    // Alternative layout based on NVIDIA m16n8 fragment pattern:
    // Thread groupNum = lane_id / 4 (0-7)
    // Thread groupIdx = lane_id % 4 (0-3)
    //   d[0]: row = groupNum,     col = groupIdx * 2
    //   d[1]: row = groupNum,     col = groupIdx * 2 + 1
    //   d[2]: row = groupNum + 8, col = groupIdx * 2
    //   d[3]: row = groupNum + 8, col = groupIdx * 2 + 1

    const int warp_row_out = warp_m * WARP_M;
    const int warp_col_base_out = warp_n * WARP_N;

    #pragma unroll
    for (int ni = 0; ni < 4; ni++) {
        int n_base = warp_col_base_out + ni * MMA_N;

        // Output layout for m16n8k32 (based on empirical testing):
        // groupNum = lane_id / 4 (0-7), groupIdx = lane_id % 4 (0-3)
        //   d[0]: row = groupNum,     col = groupIdx * 2
        //   d[1]: row = groupNum,     col = groupIdx * 2 + 1
        //   d[2]: row = groupNum + 8, col = groupIdx * 2
        //   d[3]: row = groupNum + 8, col = groupIdx * 2 + 1

        int groupNum = lane_id / 4;
        int groupIdx = lane_id % 4;

        int out_rows[4] = {
            warp_row_out + groupNum,
            warp_row_out + groupNum,
            warp_row_out + groupNum + 8,
            warp_row_out + groupNum + 8
        };
        int out_cols[4] = {
            n_base + groupIdx * 2,
            n_base + groupIdx * 2 + 1,
            n_base + groupIdx * 2,
            n_base + groupIdx * 2 + 1
        };

        #pragma unroll
        for (int ri = 0; ri < 4; ri++) {
            int global_row = block_b + out_rows[ri];
            int global_col = block_n + out_cols[ri];

            if (global_row < B && global_col < N) {
                float s_x = row_scale[out_rows[ri]];
                float w_scale = static_cast<float>(scale[global_col]);
                float b_val = (bias != nullptr) ? static_cast<float>(bias[global_col]) : 0.0f;
                // y = acc * s_x / 127 * w_scale + bias
                float result = (static_cast<float>(acc[ni][ri]) * s_x / 127.0f) * w_scale + b_val;
                Y[global_row * N + global_col] = static_cast<scalar_t>(result);
            }
        }
    }
}

//=============================================================================
// DISPATCH FUNCTION
//=============================================================================

torch::Tensor ternary_gemm_tensor_core_cuda(
    torch::Tensor X,          // [B, K] float32
    torch::Tensor W_T,        // [K_bytes, N] uint8 packed
    torch::Tensor scale,      // [N] float32
    c10::optional<torch::Tensor> bias,
    int64_t in_features,      // K (for validation)
    int64_t version           // 1 = V1 (float compute), 2 = V2 (mma.sync m16n8k32)
) {
    // Initialize LUT if needed
    init_unpack_lut();

    // Validate inputs
    TORCH_CHECK(X.dim() == 2, "X must be 2D");
    TORCH_CHECK(W_T.dim() == 2, "W_T must be 2D");
    TORCH_CHECK(W_T.dtype() == torch::kUInt8, "W_T must be uint8");
    TORCH_CHECK(version == 1 || version == 2, "version must be 1 (V1 float) or 2 (V2 mma.sync)");

    const int B = X.size(0);
    const int K = X.size(1);
    const int K_bytes = W_T.size(0);
    const int N = W_T.size(1);

    TORCH_CHECK(K_bytes == ceildiv(K, 4), "K_bytes doesn't match ceil(K/4)");
    TORCH_CHECK(scale.size(0) == N, "scale size must match N");

    // Create output tensor
    auto Y = torch::empty({B, N}, X.options());

    // Get bias pointer (nullptr if not provided)
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;

    // Launch kernel based on version
    dim3 block(THREADS_X, THREADS_Y);

    if (version == 1) {
        // V1: LUT unpack + float compute (baseline)
        dim3 grid(ceildiv(B, BLOCK_M), ceildiv(N, BLOCK_N));
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(X.scalar_type(), "ternary_gemm_tensor_core_v1", [&] {
            kernel_v1_lut_unpack<scalar_t><<<grid, block>>>(
                X.data_ptr<scalar_t>(),
                W_T.data_ptr<uint8_t>(),
                scale.data_ptr<scalar_t>(),
                bias_ptr ? reinterpret_cast<const scalar_t*>(bias_ptr) : nullptr,
                Y.data_ptr<scalar_t>(),
                B, N, K, K_bytes
            );
        });
    } else {
        // V2: LUT unpack + INT8 mma.sync (Tensor Cores)
        // Requires K to be multiple of 32 for best performance
        dim3 grid(ceildiv(B, V2_BLOCK_M), ceildiv(N, V2_BLOCK_N));
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(X.scalar_type(), "ternary_gemm_tensor_core_v2", [&] {
            kernel_v2_tensor_core<scalar_t><<<grid, block>>>(
                X.data_ptr<scalar_t>(),
                W_T.data_ptr<uint8_t>(),
                scale.data_ptr<scalar_t>(),
                bias_ptr ? reinterpret_cast<const scalar_t*>(bias_ptr) : nullptr,
                Y.data_ptr<scalar_t>(),
                B, N, K, K_bytes
            );
        });
    }

    return Y;
}

} // namespace tensor_core
} // namespace bittorch
