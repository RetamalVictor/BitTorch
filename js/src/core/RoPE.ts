/**
 * RoPE - Rotary Position Embeddings
 *
 * Implements rotary position encoding for attention mechanism.
 *
 * @module core/RoPE
 */

/** Default RoPE theta value */
const ROPE_THETA = 10000.0;

/**
 * RoPE cache containing precomputed cos/sin values.
 */
export interface RoPECache {
  /** Cosine values: [maxSeqLen, headDim/2] */
  cos: Float32Array;
  /** Sine values: [maxSeqLen, headDim/2] */
  sin: Float32Array;
  /** Head dimension */
  headDim: number;
}

/**
 * Initialize RoPE cos/sin cache.
 *
 * @param maxSeqLen - Maximum sequence length
 * @param headDim - Attention head dimension
 * @param theta - RoPE theta parameter (default: 10000)
 * @returns RoPE cache with precomputed values
 */
export function initRoPECache(
  maxSeqLen: number,
  headDim: number,
  theta: number = ROPE_THETA
): RoPECache {
  const halfDim = headDim / 2;

  const cos = new Float32Array(maxSeqLen * halfDim);
  const sin = new Float32Array(maxSeqLen * halfDim);

  for (let pos = 0; pos < maxSeqLen; pos++) {
    for (let i = 0; i < halfDim; i++) {
      const freq = 1.0 / Math.pow(theta, (2 * i) / headDim);
      const angle = pos * freq;
      cos[pos * halfDim + i] = Math.cos(angle);
      sin[pos * halfDim + i] = Math.sin(angle);
    }
  }

  return { cos, sin, headDim };
}

/**
 * Apply RoPE to Q or K tensor.
 *
 * Modifies tensor in place.
 *
 * @param x - Q or K tensor [seqLen, nHeads * headDim]
 * @param cache - Precomputed RoPE cache
 * @param seqLen - Sequence length
 * @param nHeads - Number of attention heads
 * @param dim - Model dimension (nHeads * headDim)
 */
export function applyRoPE(
  x: Float32Array,
  cache: RoPECache,
  seqLen: number,
  nHeads: number,
  dim: number
): void {
  const { cos, sin, headDim } = cache;
  const halfDim = headDim / 2;

  for (let t = 0; t < seqLen; t++) {
    for (let h = 0; h < nHeads; h++) {
      for (let i = 0; i < halfDim; i++) {
        const idx = t * dim + h * headDim + i;
        const idx2 = idx + halfDim;

        const cosVal = cos[t * halfDim + i];
        const sinVal = sin[t * halfDim + i];

        const x0 = x[idx];
        const x1 = x[idx2];

        x[idx] = x0 * cosVal - x1 * sinVal;
        x[idx2] = x0 * sinVal + x1 * cosVal;
      }
    }
  }
}

/**
 * Apply RoPE to single position Q tensor.
 *
 * Modifies tensor in place.
 *
 * @param q - Q tensor [nHeads * headDim]
 * @param cache - Precomputed RoPE cache
 * @param pos - Position index
 * @param nHeads - Number of attention heads
 */
export function applyRoPESingle(
  q: Float32Array,
  cache: RoPECache,
  pos: number,
  nHeads: number
): void {
  const { cos, sin, headDim } = cache;
  const halfDim = headDim / 2;

  for (let h = 0; h < nHeads; h++) {
    for (let i = 0; i < halfDim; i++) {
      const idx = h * headDim + i;
      const idx2 = idx + halfDim;

      const cosVal = cos[pos * halfDim + i];
      const sinVal = sin[pos * halfDim + i];

      const x0 = q[idx];
      const x1 = q[idx2];

      q[idx] = x0 * cosVal - x1 * sinVal;
      q[idx2] = x0 * sinVal + x1 * cosVal;
    }
  }
}

/**
 * Apply RoPE to K values in cache at all positions.
 *
 * @param k - K cache [nKvHeads, maxSeqLen, headDim]
 * @param cache - Precomputed RoPE cache
 * @param seqLen - Current sequence length
 * @param nKvHeads - Number of KV heads
 * @param maxSeqLen - Maximum sequence length
 */
export function applyRoPEToCache(
  k: Float32Array,
  cache: RoPECache,
  seqLen: number,
  nKvHeads: number,
  maxSeqLen: number
): void {
  const { cos, sin, headDim } = cache;
  const halfDim = headDim / 2;

  for (let t = 0; t < seqLen; t++) {
    for (let kh = 0; kh < nKvHeads; kh++) {
      const baseIdx = kh * maxSeqLen * headDim + t * headDim;
      for (let i = 0; i < halfDim; i++) {
        const idx = baseIdx + i;
        const idx2 = idx + halfDim;

        const cosVal = cos[t * halfDim + i];
        const sinVal = sin[t * halfDim + i];

        const x0 = k[idx];
        const x1 = k[idx2];

        k[idx] = x0 * cosVal - x1 * sinVal;
        k[idx2] = x0 * sinVal + x1 * cosVal;
      }
    }
  }
}

/**
 * Apply RoPE to K values at single position in cache.
 *
 * @param k - K cache [nKvHeads, maxSeqLen, headDim]
 * @param cache - Precomputed RoPE cache
 * @param pos - Position index
 * @param nKvHeads - Number of KV heads
 * @param maxSeqLen - Maximum sequence length
 */
export function applyRoPEToSinglePos(
  k: Float32Array,
  cache: RoPECache,
  pos: number,
  nKvHeads: number,
  maxSeqLen: number
): void {
  const { cos, sin, headDim } = cache;
  const halfDim = headDim / 2;

  for (let kh = 0; kh < nKvHeads; kh++) {
    const baseIdx = kh * maxSeqLen * headDim + pos * headDim;
    for (let i = 0; i < halfDim; i++) {
      const idx = baseIdx + i;
      const idx2 = idx + halfDim;

      const cosVal = cos[pos * halfDim + i];
      const sinVal = sin[pos * halfDim + i];

      const x0 = k[idx];
      const x1 = k[idx2];

      k[idx] = x0 * cosVal - x1 * sinVal;
      k[idx2] = x0 * sinVal + x1 * cosVal;
    }
  }
}
