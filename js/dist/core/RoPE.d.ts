/**
 * RoPE - Rotary Position Embeddings
 *
 * Implements rotary position encoding for attention mechanism.
 *
 * @module core/RoPE
 */
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
export declare function initRoPECache(maxSeqLen: number, headDim: number, theta?: number): RoPECache;
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
export declare function applyRoPE(x: Float32Array, cache: RoPECache, seqLen: number, nHeads: number, dim: number): void;
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
export declare function applyRoPESingle(q: Float32Array, cache: RoPECache, pos: number, nHeads: number): void;
/**
 * Apply RoPE to K values in cache at all positions.
 *
 * @param k - K cache [nKvHeads, maxSeqLen, headDim]
 * @param cache - Precomputed RoPE cache
 * @param seqLen - Current sequence length
 * @param nKvHeads - Number of KV heads
 * @param maxSeqLen - Maximum sequence length
 */
export declare function applyRoPEToCache(k: Float32Array, cache: RoPECache, seqLen: number, nKvHeads: number, maxSeqLen: number): void;
/**
 * Apply RoPE to K values at single position in cache.
 *
 * @param k - K cache [nKvHeads, maxSeqLen, headDim]
 * @param cache - Precomputed RoPE cache
 * @param pos - Position index
 * @param nKvHeads - Number of KV heads
 * @param maxSeqLen - Maximum sequence length
 */
export declare function applyRoPEToSinglePos(k: Float32Array, cache: RoPECache, pos: number, nKvHeads: number, maxSeqLen: number): void;
//# sourceMappingURL=RoPE.d.ts.map