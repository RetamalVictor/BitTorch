/**
 * RMSNorm - Root Mean Square Layer Normalization
 *
 * Implements: output = x * weight / sqrt(mean(x²) + eps)
 *
 * @module core/RMSNorm
 */

/** RMSNorm epsilon for numerical stability */
const NORM_EPS = 1e-5;

/**
 * RMSNorm for a batch of vectors.
 *
 * @param x - Input tensor [seqLen, dim]
 * @param weight - Scale weights [dim]
 * @param seqLen - Number of vectors
 * @param dim - Vector dimension
 * @returns Normalized tensor [seqLen, dim]
 */
export function rmsNorm(
  x: Float32Array,
  weight: Float32Array,
  seqLen: number,
  dim: number
): Float32Array {
  const result = new Float32Array(x.length);

  for (let t = 0; t < seqLen; t++) {
    const offset = t * dim;

    // Compute RMS
    let sumSq = 0;
    for (let d = 0; d < dim; d++) {
      const val = x[offset + d];
      sumSq += val * val;
    }
    const rms = Math.sqrt(sumSq / dim + NORM_EPS);

    // Normalize and scale
    for (let d = 0; d < dim; d++) {
      result[offset + d] = (x[offset + d] / rms) * weight[d];
    }
  }

  return result;
}

/**
 * RMSNorm for a single vector.
 *
 * Optimized path for decode phase (seqLen=1).
 *
 * @param x - Input vector [dim]
 * @param weight - Scale weights [dim]
 * @returns Normalized vector [dim]
 */
export function rmsNormSingle(
  x: Float32Array,
  weight: Float32Array
): Float32Array {
  const dim = x.length;
  const result = new Float32Array(dim);

  // Compute RMS
  let sumSq = 0;
  for (let d = 0; d < dim; d++) {
    sumSq += x[d] * x[d];
  }
  const rms = Math.sqrt(sumSq / dim + NORM_EPS);

  // Normalize and scale
  for (let d = 0; d < dim; d++) {
    result[d] = (x[d] / rms) * weight[d];
  }

  return result;
}
