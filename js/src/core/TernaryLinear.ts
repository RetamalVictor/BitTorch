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
export function ternaryMatmul(
  input: Float32Array,
  layer: TernaryLayer,
  seqLen: number
): Float32Array {
  const { inFeatures, outFeatures, weightsPacked, scales } = layer;
  const output = new Float32Array(seqLen * outFeatures);
  const inBytes = Math.ceil(inFeatures / 4);
  const fullBytes = Math.floor(inFeatures / 4);

  for (let t = 0; t < seqLen; t++) {
    const inputOffset = t * inFeatures;
    const outputOffset = t * outFeatures;

    for (let n = 0; n < outFeatures; n++) {
      let acc = 0;
      const weightOffset = n * inBytes;

      // Process full bytes - unrolled with branch-free arithmetic
      for (let kb = 0; kb < fullBytes; kb++) {
        const packed = weightsPacked[weightOffset + kb];
        const k = inputOffset + kb * 4;

        // Extract codes and decode branch-free: w = (code & 1) - (code >> 1)
        const c0 = packed & 0x3;
        const c1 = (packed >> 2) & 0x3;
        const c2 = (packed >> 4) & 0x3;
        const c3 = (packed >> 6) & 0x3;

        acc += input[k] * ((c0 & 1) - (c0 >> 1));
        acc += input[k + 1] * ((c1 & 1) - (c1 >> 1));
        acc += input[k + 2] * ((c2 & 1) - (c2 >> 1));
        acc += input[k + 3] * ((c3 & 1) - (c3 >> 1));
      }

      // Handle remaining elements (if inFeatures not divisible by 4)
      if (fullBytes < inBytes) {
        const packed = weightsPacked[weightOffset + fullBytes];
        const remaining = inFeatures - fullBytes * 4;
        const k = inputOffset + fullBytes * 4;

        for (let i = 0; i < remaining; i++) {
          const code = (packed >> (i * 2)) & 0x3;
          acc += input[k + i] * ((code & 1) - (code >> 1));
        }
      }

      output[outputOffset + n] = acc * scales[n];
    }
  }

  return output;
}

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
export function ternaryMatmulSingle(
  input: Float32Array,
  layer: TernaryLayer
): Float32Array {
  const { inFeatures, outFeatures, weightsPacked, scales } = layer;
  const output = new Float32Array(outFeatures);
  const inBytes = Math.ceil(inFeatures / 4);
  const fullBytes = Math.floor(inFeatures / 4);

  for (let n = 0; n < outFeatures; n++) {
    let acc = 0;
    const weightOffset = n * inBytes;

    // Process full bytes - unrolled with branch-free arithmetic
    for (let kb = 0; kb < fullBytes; kb++) {
      const packed = weightsPacked[weightOffset + kb];
      const k = kb * 4;

      const c0 = packed & 0x3;
      const c1 = (packed >> 2) & 0x3;
      const c2 = (packed >> 4) & 0x3;
      const c3 = (packed >> 6) & 0x3;

      acc += input[k] * ((c0 & 1) - (c0 >> 1));
      acc += input[k + 1] * ((c1 & 1) - (c1 >> 1));
      acc += input[k + 2] * ((c2 & 1) - (c2 >> 1));
      acc += input[k + 3] * ((c3 & 1) - (c3 >> 1));
    }

    // Handle remaining elements
    if (fullBytes < inBytes) {
      const packed = weightsPacked[weightOffset + fullBytes];
      const remaining = inFeatures - fullBytes * 4;
      const k = fullBytes * 4;

      for (let i = 0; i < remaining; i++) {
        const code = (packed >> (i * 2)) & 0x3;
        acc += input[k + i] * ((code & 1) - (code >> 1));
      }
    }

    output[n] = acc * scales[n];
  }

  return output;
}

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
export function ternaryMatmulSingleInto(
  input: Float32Array,
  layer: TernaryLayer,
  output: Float32Array
): void {
  const { inFeatures, outFeatures, weightsPacked, scales } = layer;
  const inBytes = Math.ceil(inFeatures / 4);
  const fullBytes = Math.floor(inFeatures / 4);

  for (let n = 0; n < outFeatures; n++) {
    let acc = 0;
    const weightOffset = n * inBytes;

    // Process full bytes - unrolled with branch-free arithmetic
    for (let kb = 0; kb < fullBytes; kb++) {
      const packed = weightsPacked[weightOffset + kb];
      const k = kb * 4;

      const c0 = packed & 0x3;
      const c1 = (packed >> 2) & 0x3;
      const c2 = (packed >> 4) & 0x3;
      const c3 = (packed >> 6) & 0x3;

      acc += input[k] * ((c0 & 1) - (c0 >> 1));
      acc += input[k + 1] * ((c1 & 1) - (c1 >> 1));
      acc += input[k + 2] * ((c2 & 1) - (c2 >> 1));
      acc += input[k + 3] * ((c3 & 1) - (c3 >> 1));
    }

    // Handle remaining elements
    if (fullBytes < inBytes) {
      const packed = weightsPacked[weightOffset + fullBytes];
      const remaining = inFeatures - fullBytes * 4;
      const k = fullBytes * 4;

      for (let i = 0; i < remaining; i++) {
        const code = (packed >> (i * 2)) & 0x3;
        acc += input[k + i] * ((code & 1) - (code >> 1));
      }
    }

    output[n] = acc * scales[n];
  }
}

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
export function matmulFP32(
  input: Float32Array,
  weights: Float32Array,
  K: number,
  N: number
): Float32Array {
  const output = new Float32Array(N);

  for (let n = 0; n < N; n++) {
    let acc = 0;
    const weightOffset = n * K;
    for (let k = 0; k < K; k++) {
      acc += input[k] * weights[weightOffset + k];
    }
    output[n] = acc;
  }

  return output;
}

/**
 * FP32 matrix multiplication - in-place version.
 *
 * @param input - Input vector [K]
 * @param weights - Weight matrix [N, K]
 * @param K - Input dimension
 * @param N - Output dimension
 * @param output - Pre-allocated output buffer [N]
 */
export function matmulFP32Into(
  input: Float32Array,
  weights: Float32Array,
  K: number,
  N: number,
  output: Float32Array
): void {
  for (let n = 0; n < N; n++) {
    let acc = 0;
    const weightOffset = n * K;
    for (let k = 0; k < K; k++) {
      acc += input[k] * weights[weightOffset + k];
    }
    output[n] = acc;
  }
}
