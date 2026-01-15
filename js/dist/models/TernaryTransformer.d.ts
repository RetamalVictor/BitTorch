/**
 * TernaryTransformer - LLaMA-style transformer with ternary quantization
 *
 * Main model class for browser-based inference with ternary quantized weights.
 *
 * @module models/TernaryTransformer
 */
import type { TransformerConfig, GenerateOptions, MemoryStats } from "../core/types.js";
/**
 * TernaryTransformer - LLaMA-style transformer with ternary quantization.
 *
 * @example
 * ```typescript
 * const model = await TernaryTransformer.load('/path/to/model/');
 *
 * const output = await model.generate('Hello, ', {
 *   maxTokens: 50,
 *   temperature: 0.8,
 *   onToken: (token, stats) => console.log(token)
 * });
 *
 * model.destroy();
 * ```
 */
export declare class TernaryTransformer {
    private _config;
    private tokenizer;
    private embedding;
    private blocks;
    private normWeight;
    private head;
    private ropeCache;
    private kvCache;
    private cacheSeqLen;
    private gpu;
    private stopRequested;
    private ternaryWeights;
    private fp16Weights;
    private decodeBuffers;
    private constructor();
    /**
     * Load model from a directory containing config.json, tokenizer.json, and model.safetensors.
     *
     * @param basePath - URL or path to model directory
     * @returns Promise resolving to TernaryTransformer instance
     */
    static load(basePath: string): Promise<TernaryTransformer>;
    /**
     * Initialize pre-allocated buffers for decode (called once after load).
     */
    private initDecodeBuffers;
    /**
     * Get model configuration.
     */
    get config(): TransformerConfig;
    /**
     * Check if GPU acceleration is enabled.
     */
    get isGPUEnabled(): boolean;
    /**
     * Get memory usage statistics.
     */
    get memoryStats(): MemoryStats;
    /**
     * Generate text continuation.
     *
     * @param prompt - Starting text
     * @param options - Generation options
     * @returns Generated text
     */
    generate(prompt: string, options?: GenerateOptions): Promise<string>;
    /**
     * Stop ongoing generation.
     */
    stop(): void;
    /**
     * Release all resources.
     */
    destroy(): void;
    private initKVCache;
    private forwardPrefill;
    private forwardDecode;
    private ternaryMatmulDispatch;
    private swiglu;
    /**
     * Optimized SwiGLU MLP using pre-allocated buffers.
     * Writes result to buf.mlpOut.
     */
    private swigluSingleOptimized;
    /**
     * Optimized attention decode using pre-allocated buffers.
     * Writes result to buf.attnOut.
     */
    private attentionDecodeOptimized;
    private attentionPrefill;
    private sample;
}
//# sourceMappingURL=TernaryTransformer.d.ts.map