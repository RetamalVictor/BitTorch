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
  [key: string]: TensorInfo | { __metadata__?: Record<string, string> };
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
export class SafeTensorsLoader {
  private buffer: ArrayBuffer;
  private header: SafeTensorsHeader;
  private dataOffset: number;

  private constructor(
    buffer: ArrayBuffer,
    header: SafeTensorsHeader,
    dataOffset: number
  ) {
    this.buffer = buffer;
    this.header = header;
    this.dataOffset = dataOffset;
  }

  /**
   * Load SafeTensors from a URL.
   *
   * @param url - URL to the .safetensors file
   * @returns Promise resolving to SafeTensorsLoader instance
   * @throws Error if fetch fails
   */
  static async fromUrl(url: string): Promise<SafeTensorsLoader> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch ${url}: ${response.statusText}`);
    }
    const buffer = await response.arrayBuffer();
    return SafeTensorsLoader.fromBuffer(buffer);
  }

  /**
   * Load SafeTensors from an ArrayBuffer.
   *
   * @param buffer - Raw file contents
   * @returns SafeTensorsLoader instance
   * @throws Error if file format is invalid
   */
  static fromBuffer(buffer: ArrayBuffer): SafeTensorsLoader {
    // Read header size (first 8 bytes, little-endian u64)
    const view = new DataView(buffer);
    const headerSizeLow = view.getUint32(0, true);
    const headerSizeHigh = view.getUint32(4, true);

    // JavaScript can't handle u64 natively, but header should be < 4GB
    if (headerSizeHigh !== 0) {
      throw new Error("Header size too large (> 4GB)");
    }
    const headerSize = headerSizeLow;

    // Parse header JSON
    const headerBytes = new Uint8Array(buffer, 8, headerSize);
    const headerStr = new TextDecoder().decode(headerBytes);
    const header = JSON.parse(headerStr) as SafeTensorsHeader;

    const dataOffset = 8 + headerSize;

    return new SafeTensorsLoader(buffer, header, dataOffset);
  }

  /**
   * Get list of tensor names in the file.
   *
   * @returns Array of tensor names (excludes __metadata__)
   */
  getTensorNames(): string[] {
    return Object.keys(this.header).filter((k) => k !== "__metadata__");
  }

  /**
   * Check if a tensor exists in the file.
   *
   * @param name - Tensor name to check
   * @returns true if tensor exists
   */
  hasTensor(name: string): boolean {
    return name in this.header && name !== "__metadata__";
  }

  /**
   * Get tensor metadata without loading data.
   *
   * @param name - Tensor name
   * @returns TensorInfo or null if not found
   */
  getTensorInfo(name: string): TensorInfo | null {
    const info = this.header[name];
    if (!info || name === "__metadata__") return null;
    return info as TensorInfo;
  }

  /**
   * Load a tensor as raw ArrayBuffer.
   *
   * @param name - Tensor name
   * @returns LoadedTensor with raw data
   * @throws Error if tensor not found
   */
  getTensorBuffer(name: string): LoadedTensor {
    const info = this.header[name] as TensorInfo & {
      data_offsets?: [number, number];
    };
    if (!info) {
      throw new Error(`Tensor not found: ${name}`);
    }

    // Handle both camelCase and snake_case (SafeTensors uses snake_case)
    const dataOffsets = info.dataOffsets || info.data_offsets;
    if (!dataOffsets) {
      throw new Error(`Tensor ${name} has no data_offsets`);
    }

    const [start, end] = dataOffsets;
    const data = this.buffer.slice(
      this.dataOffset + start,
      this.dataOffset + end
    );

    return {
      data,
      dtype: info.dtype,
      shape: info.shape,
    };
  }

  /**
   * Load tensor as Float32Array.
   *
   * Converts from FP16/BF16 if needed.
   *
   * @param name - Tensor name
   * @returns Float32Array with tensor data and shape
   * @throws Error if tensor not found or unsupported dtype
   */
  getTensorFloat32(name: string): { data: Float32Array; shape: number[] } {
    const tensor = this.getTensorBuffer(name);

    if (tensor.dtype === "F32") {
      return {
        data: new Float32Array(tensor.data),
        shape: tensor.shape,
      };
    } else if (tensor.dtype === "F16") {
      const f16 = new Uint16Array(tensor.data);
      const f32 = new Float32Array(f16.length);
      for (let i = 0; i < f16.length; i++) {
        f32[i] = float16ToFloat32(f16[i]);
      }
      return {
        data: f32,
        shape: tensor.shape,
      };
    } else if (tensor.dtype === "BF16") {
      const bf16 = new Uint16Array(tensor.data);
      const f32 = new Float32Array(bf16.length);
      for (let i = 0; i < bf16.length; i++) {
        f32[i] = bfloat16ToFloat32(bf16[i]);
      }
      return {
        data: f32,
        shape: tensor.shape,
      };
    }

    throw new Error(`Unsupported dtype for float conversion: ${tensor.dtype}`);
  }

  /**
   * Load tensor as Uint8Array (for packed ternary weights).
   *
   * @param name - Tensor name
   * @returns Uint8Array with tensor data and shape
   * @throws Error if tensor not found or dtype is not U8
   */
  getTensorUint8(name: string): { data: Uint8Array; shape: number[] } {
    const tensor = this.getTensorBuffer(name);

    if (tensor.dtype !== "U8") {
      throw new Error(`Expected U8 dtype, got ${tensor.dtype}`);
    }

    return {
      data: new Uint8Array(tensor.data),
      shape: tensor.shape,
    };
  }

  /**
   * Get metadata if present in the file.
   *
   * @returns Metadata record or null if not present
   */
  getMetadata(): Record<string, string> | null {
    const meta = this.header["__metadata__"];
    if (meta && typeof meta === "object" && !("dataOffsets" in meta)) {
      return meta as Record<string, string>;
    }
    return null;
  }

  /**
   * Get the total size of the file in bytes.
   */
  get byteLength(): number {
    return this.buffer.byteLength;
  }
}

/**
 * Convert FP16 (IEEE 754 half-precision) to FP32.
 */
function float16ToFloat32(h: number): number {
  const sign = (h & 0x8000) >> 15;
  const exp = (h & 0x7c00) >> 10;
  const frac = h & 0x03ff;

  if (exp === 0) {
    // Subnormal or zero
    if (frac === 0) return sign ? -0 : 0;
    const e = -14;
    const m = frac / 1024;
    return (sign ? -1 : 1) * m * Math.pow(2, e);
  } else if (exp === 31) {
    // Infinity or NaN
    if (frac === 0) return sign ? -Infinity : Infinity;
    return NaN;
  }

  const e = exp - 15;
  const m = 1 + frac / 1024;
  return (sign ? -1 : 1) * m * Math.pow(2, e);
}

/**
 * Convert BF16 to FP32.
 * BF16 is just the upper 16 bits of FP32.
 */
function bfloat16ToFloat32(bf16: number): number {
  const buffer = new ArrayBuffer(4);
  const view = new DataView(buffer);
  // BF16 goes into upper 16 bits
  view.setUint16(2, bf16, false); // Big-endian for upper bytes
  view.setUint16(0, 0, false); // Lower 16 bits are zero
  return view.getFloat32(0, false);
}
