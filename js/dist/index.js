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
// Main model class
export { TernaryTransformer } from "./models/TernaryTransformer.js";
// Tokenizer
export { BPETokenizer } from "./tokenizers/BPETokenizer.js";
// Loaders
export { SafeTensorsLoader } from "./loaders/SafeTensorsLoader.js";
// Core functions (for advanced use)
export { ternaryMatmul, ternaryMatmulSingle, matmulFP32 } from "./core/TernaryLinear.js";
export { rmsNorm, rmsNormSingle } from "./core/RMSNorm.js";
export { embeddingLookup, embeddingLookupSingle } from "./core/Embedding.js";
export { initRoPECache, applyRoPE, applyRoPESingle } from "./core/RoPE.js";
// GPU backend (for advanced use)
export { WebGPUBackend } from "./gpu/WebGPUBackend.js";
//# sourceMappingURL=index.js.map