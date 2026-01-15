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

  for (let t = 0; t < seqLen; t++) {
    const inputOffset = t * inFeatures;
    const outputOffset = t * outFeatures;

    for (let n = 0; n < outFeatures; n++) {
      let acc = 0;
      const weightOffset = n * inBytes;

      // Process packed weights
      for (let kb = 0; kb < inBytes; kb++) {
        const packed = weightsPacked[weightOffset + kb];

        // Unpack 4 ternary values from byte
        for (let i = 0; i < 4 && kb * 4 + i < inFeatures; i++) {
          const code = (packed >> (i * 2)) & 0x3;
          // 0 = 0, 1 = +1, 2 = -1
          const w = code === 1 ? 1 : code === 2 ? -1 : 0;
          const k = kb * 4 + i;
          acc += input[inputOffset + k] * w;
        }
      }

      // Apply scale
      output[outputOffset + n] = acc * scales[n];
    }
  }

  return output;
}

/**
 * Ternary matrix multiplication for single vector (CPU).
 *
 * Optimized path for decode phase (seqLen=1).
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

  for (let n = 0; n < outFeatures; n++) {
    let acc = 0;
    const weightOffset = n * inBytes;

    // Process packed weights
    for (let kb = 0; kb < inBytes; kb++) {
      const packed = weightsPacked[weightOffset + kb];

      // Unpack 4 ternary values from byte
      for (let i = 0; i < 4 && kb * 4 + i < inFeatures; i++) {
        const code = (packed >> (i * 2)) & 0x3;
        const w = code === 1 ? 1 : code === 2 ? -1 : 0;
        const k = kb * 4 + i;
        acc += input[k] * w;
      }
    }

    output[n] = acc * scales[n];
  }

  return output;
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
