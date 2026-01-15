/**
 * @bittorch/js - JavaScript/WebGPU inference library for ternary-quantized transformers
 *
 * Provides browser-based inference for LLaMA-style transformers with 2-bit ternary weights.
 *
 * @example
 * ```typescript
 * import { TernaryTransformer } from '@bittorch/js';
 *
 * const model = await TernaryTransformer.load('/path/to/model/');
 *
 * const output = await model.generate('Hello, ', {
 *   maxTokens: 50,
 *   temperature: 0.8,
 *   onToken: (token, stats) => console.log(token)
 * });
 *
 * console.log(model.config);      // { vocabSize, dim, nLayers, ... }
 * console.log(model.memoryStats); // { packedKB, fp16EquivalentKB, ... }
 * console.log(model.isGPUEnabled);
 *
 * model.destroy();
 * ```
 *
 * @module @bittorch/js
 */
export { TernaryTransformer } from "./models/TernaryTransformer.js";
export { BPETokenizer } from "./tokenizers/BPETokenizer.js";
export { SafeTensorsLoader } from "./loaders/SafeTensorsLoader.js";
export type { TernaryLayer, TransformerConfig, GenerationStats, GenerateOptions, MemoryStats, } from "./core/types.js";
export { ternaryMatmul, ternaryMatmulSingle, matmulFP32 } from "./core/TernaryLinear.js";
export { rmsNorm, rmsNormSingle } from "./core/RMSNorm.js";
export { embeddingLookup, embeddingLookupSingle } from "./core/Embedding.js";
export { initRoPECache, applyRoPE, applyRoPESingle, type RoPECache } from "./core/RoPE.js";
export { WebGPUBackend } from "./gpu/WebGPUBackend.js";
//# sourceMappingURL=index.d.ts.map