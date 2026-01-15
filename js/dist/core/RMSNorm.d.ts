/**
 * RMSNorm - Root Mean Square Layer Normalization
 *
 * Implements: output = x * weight / sqrt(mean(x²) + eps)
 *
 * @module core/RMSNorm
 */
/**
 * RMSNorm for a batch of vectors.
 *
 * @param x - Input tensor [seqLen, dim]
 * @param weight - Scale weights [dim]
 * @param seqLen - Number of vectors
 * @param dim - Vector dimension
 * @returns Normalized tensor [seqLen, dim]
 */
export declare function rmsNorm(x: Float32Array, weight: Float32Array, seqLen: number, dim: number): Float32Array;
/**
 * RMSNorm for a single vector.
 *
 * Optimized path for decode phase (seqLen=1).
 *
 * @param x - Input vector [dim]
 * @param weight - Scale weights [dim]
 * @returns Normalized vector [dim]
 */
export declare function rmsNormSingle(x: Float32Array, weight: Float32Array): Float32Array;
/**
 * RMSNorm for a single vector - in-place version.
 *
 * @param x - Input vector [dim]
 * @param weight - Scale weights [dim]
 * @param output - Pre-allocated output buffer [dim]
 */
export declare function rmsNormSingleInto(x: Float32Array, weight: Float32Array, output: Float32Array): void;
//# sourceMappingURL=RMSNorm.d.ts.map