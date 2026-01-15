/**
 * SafeTensors Loader for Browser
 *
 * Parses the SafeTensors format (HuggingFace standard) in JavaScript.
 * https://huggingface.co/docs/safetensors
 *
 * @module loaders/SafeTensorsLoader
 */
/** Tensor metadata from SafeTensors header */
export interface TensorInfo {
    dtype: string;
    shape: number[];
    dataOffsets: [number, number];
}
/** SafeTensors file header structure */
export interface SafeTensorsHeader {
    [key: string]: TensorInfo | {
        __metadata__?: Record<string, string>;
    };
}
/** Loaded tensor with raw data */
export interface LoadedTensor {
    data: ArrayBuffer;
    dtype: string;
    shape: number[];
}
/**
 * SafeTensors file loader.
 *
 * Loads and parses SafeTensors format files, which is the HuggingFace
 * standard for storing model weights safely and efficiently.
 *
 * @example
 * ```typescript
 * const loader = await SafeTensorsLoader.fromUrl('/model/weights.safetensors');
 *
 * // List available tensors
 * console.log(loader.getTensorNames());
 *
 * // Load a tensor as Float32Array
 * const { data, shape } = loader.getTensorFloat32('embedding.weight');
 *
 * // Load packed ternary weights as Uint8Array
 * const packed = loader.getTensorUint8('layer.weight_packed');
 * ```
 */
export declare class SafeTensorsLoader {
    private buffer;
    private header;
    private dataOffset;
    private constructor();
    /**
     * Load SafeTensors from a URL.
     *
     * @param url - URL to the .safetensors file
     * @returns Promise resolving to SafeTensorsLoader instance
     * @throws Error if fetch fails
     */
    static fromUrl(url: string): Promise<SafeTensorsLoader>;
    /**
     * Load SafeTensors from an ArrayBuffer.
     *
     * @param buffer - Raw file contents
     * @returns SafeTensorsLoader instance
     * @throws Error if file format is invalid
     */
    static fromBuffer(buffer: ArrayBuffer): SafeTensorsLoader;
    /**
     * Get list of tensor names in the file.
     *
     * @returns Array of tensor names (excludes __metadata__)
     */
    getTensorNames(): string[];
    /**
     * Check if a tensor exists in the file.
     *
     * @param name - Tensor name to check
     * @returns true if tensor exists
     */
    hasTensor(name: string): boolean;
    /**
     * Get tensor metadata without loading data.
     *
     * @param name - Tensor name
     * @returns TensorInfo or null if not found
     */
    getTensorInfo(name: string): TensorInfo | null;
    /**
     * Load a tensor as raw ArrayBuffer.
     *
     * @param name - Tensor name
     * @returns LoadedTensor with raw data
     * @throws Error if tensor not found
     */
    getTensorBuffer(name: string): LoadedTensor;
    /**
     * Load tensor as Float32Array.
     *
     * Converts from FP16/BF16 if needed.
     *
     * @param name - Tensor name
     * @returns Float32Array with tensor data and shape
     * @throws Error if tensor not found or unsupported dtype
     */
    getTensorFloat32(name: string): {
        data: Float32Array;
        shape: number[];
    };
    /**
     * Load tensor as Uint8Array (for packed ternary weights).
     *
     * @param name - Tensor name
     * @returns Uint8Array with tensor data and shape
     * @throws Error if tensor not found or dtype is not U8
     */
    getTensorUint8(name: string): {
        data: Uint8Array;
        shape: number[];
    };
    /**
     * Get metadata if present in the file.
     *
     * @returns Metadata record or null if not present
     */
    getMetadata(): Record<string, string> | null;
    /**
     * Get the total size of the file in bytes.
     */
    get byteLength(): number;
}
//# sourceMappingURL=SafeTensorsLoader.d.ts.map