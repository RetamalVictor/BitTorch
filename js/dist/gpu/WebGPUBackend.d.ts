/**
 * WebGPU Backend for Ternary Matrix Multiplication
 *
 * Provides GPU-accelerated ternary matmul for transformer inference.
 *
 * @module gpu/WebGPUBackend
 */
import type { TernaryLayer } from "../core/types.js";
/**
 * WebGPU-accelerated ternary matrix multiplication.
 *
 * @example
 * ```typescript
 * const gpu = await WebGPUBackend.create();
 * if (gpu) {
 *   gpu.uploadLayer('layer1', ternaryLayer);
 *   const output = await gpu.matmul(input, 'layer1', seqLen);
 *   gpu.destroy();
 * }
 * ```
 */
export declare class WebGPUBackend {
    private device;
    private pipeline;
    private bindGroupLayout;
    private layerBuffers;
    private inputBuffer;
    private outputBuffer;
    private stagingBuffer;
    private uniformBuffer;
    private maxInputSize;
    private maxOutputSize;
    private constructor();
    /**
     * Create a WebGPU backend instance.
     *
     * @returns Backend instance or null if WebGPU is unavailable
     */
    static create(): Promise<WebGPUBackend | null>;
    /**
     * Upload a layer's weights to GPU.
     *
     * Call once per layer at initialization time.
     *
     * @param name - Unique layer identifier
     * @param layer - Ternary layer with packed weights and scales
     */
    uploadLayer(name: string, layer: TernaryLayer): void;
    /**
     * Ensure activation buffers are large enough.
     */
    private ensureBuffers;
    /**
     * Execute a ternary matrix multiplication on GPU.
     *
     * @param input - Input tensor [seqLen, inFeatures]
     * @param layerName - Name of pre-uploaded layer
     * @param seqLen - Sequence length (number of input rows)
     * @returns Output tensor [seqLen, outFeatures]
     */
    matmul(input: Float32Array, layerName: string, seqLen: number): Promise<Float32Array>;
    /**
     * Check if a layer has been uploaded.
     */
    hasLayer(name: string): boolean;
    /**
     * Get memory usage statistics.
     */
    getMemoryStats(): {
        layerBuffersKB: number;
        activationBuffersKB: number;
    };
    /**
     * Release all GPU resources.
     */
    destroy(): void;
}
//# sourceMappingURL=WebGPUBackend.d.ts.map