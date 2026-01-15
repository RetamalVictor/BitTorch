/**
 * Core type definitions for bittorch.js
 *
 * @module core/types
 */

/** Ternary layer with packed weights and per-channel scales */
export interface TernaryLayer {
  /** Packed 2-bit weights: [outFeatures, ceil(inFeatures/4)] */
  weightsPacked: Uint8Array;
  /** Per-output-channel scales: [outFeatures] */
  scales: Float32Array;
  /** Number of output features (rows) */
  outFeatures: number;
  /** Number of input features (columns) */
  inFeatures: number;
}

/** Transformer model configuration */
export interface TransformerConfig {
  vocabSize: number;
  dim: number;
  nLayers: number;
  nHeads: number;
  /** For GQA support (nKvHeads < nHeads), defaults to nHeads */
  nKvHeads: number;
  maxSeqLen: number;
}

/** Statistics during text generation */
export interface GenerationStats {
  tokensPerSecond: number;
  totalTokens: number;
  elapsedMs: number;
}

/** Options for text generation */
export interface GenerateOptions {
  /** Maximum tokens to generate */
  maxTokens?: number;
  /** Sampling temperature (default: 0.8) */
  temperature?: number;
  /** Callback for each generated token */
  onToken?: (token: string, stats: GenerationStats) => void;
}

/** Model memory statistics */
export interface MemoryStats {
  /** Size of packed ternary weights in KB */
  packedWeightsKB: number;
  /** Equivalent FP16 size in KB (for comparison) */
  fp16EquivalentKB: number;
  /** Compression ratio vs FP16 */
  compressionRatio: number;
  /** Size of scale parameters in KB */
  scalesKB: number;
}

/** KV Cache for a single layer */
export interface LayerKVCache {
  /** Key cache: [nKvHeads, maxSeqLen, headDim] */
  k: Float32Array;
  /** Value cache: [nKvHeads, maxSeqLen, headDim] */
  v: Float32Array;
}

/** Transformer block weights */
export interface TransformerBlock {
  /** Pre-attention RMSNorm weight: [dim] */
  norm1Weight: Float32Array;
  /** Query projection */
  qProj: TernaryLayer;
  /** Key+Value projection (combined) */
  kvProj: TernaryLayer;
  /** Output projection */
  proj: TernaryLayer;
  /** Pre-MLP RMSNorm weight: [dim] */
  norm2Weight: Float32Array;
  /** SwiGLU gate projection */
  wGate: TernaryLayer;
  /** SwiGLU up projection */
  wUp: TernaryLayer;
  /** SwiGLU down projection */
  wDown: TernaryLayer;
}
