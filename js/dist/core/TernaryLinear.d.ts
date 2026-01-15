/**
 * Ternary Linear Layer - CPU Implementation
 *
 * Performs matrix multiplication with 2-bit packed ternary weights.
 * Weights are stored as 4 ternary values per byte:
 *   - 00 = 0
 *   - 01 = +1
 *   - 10 = -1
 *
 * @module core/TernaryLinear
 */
import type { TernaryLayer } from "./types.js";
/**
 * Ternary matrix multiplication (CPU).
 *
 * Computes output = input @ (weights * scales).T
 *
 * Optimized with branch-free weight decode and unrolled inner loop.
 * Uses: w = (code & 1) - (code >> 1) for branch-free ternary decode.
 *
 * @param input - Input tensor [seqLen, inFeatures]
 * @param layer - Ternary layer with packed weights and scales
 * @param seqLen - Sequence length (number of rows in input)
 * @returns Output tensor [seqLen, outFeatures]
 */
export declare function ternaryMatmul(input: Float32Array, layer: TernaryLayer, seqLen: number): Float32Array;
/**
 * Ternary matrix multiplication for single vector (CPU).
 *
 * Optimized path for decode phase (seqLen=1).
 * Uses branch-free weight decode and unrolled inner loop.
 *
 * @param input - Input vector [inFeatures]
 * @param layer - Ternary layer with packed weights and scales
 * @returns Output vector [outFeatures]
 */
export declare function ternaryMatmulSingle(input: Float32Array, layer: TernaryLayer): Float32Array;
/**
 * Ternary matrix multiplication for single vector - in-place version.
 *
 * Optimized with branch-free weight decode and unrolled inner loop.
 * This is the hot path for decode - ~6.7x faster than branching version.
 *
 * @param input - Input vector [inFeatures]
 * @param layer - Ternary layer with packed weights and scales
 * @param output - Pre-allocated output buffer [outFeatures]
 */
export declare function ternaryMatmulSingleInto(input: Float32Array, layer: TernaryLayer, output: Float32Array): void;
/**
 * FP32 matrix multiplication.
 *
 * Used for embedding head (not ternary quantized).
 *
 * @param input - Input vector [K]
 * @param weights - Weight matrix [N, K]
 * @param K - Input dimension
 * @param N - Output dimension
 * @returns Output vector [N]
 */
export declare function matmulFP32(input: Float32Array, weights: Float32Array, K: number, N: number): Float32Array;
/**
 * FP32 matrix multiplication - in-place version.
 *
 * @param input - Input vector [K]
 * @param weights - Weight matrix [N, K]
 * @param K - Input dimension
 * @param N - Output dimension
 * @param output - Pre-allocated output buffer [N]
 */
export declare function matmulFP32Into(input: Float32Array, weights: Float32Array, K: number, N: number, output: Float32Array): void;
//# sourceMappingURL=TernaryLinear.d.ts.map