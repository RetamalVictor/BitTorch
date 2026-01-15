/**
 * Tests for SafeTensors loader
 */

import { describe, it, expect } from "vitest";
import { SafeTensorsLoader } from "../src/loaders/SafeTensorsLoader.js";

/**
 * Create a minimal SafeTensors buffer for testing.
 */
function createSafeTensorsBuffer(
  tensors: Record<string, { dtype: string; shape: number[]; data: ArrayBuffer }>
): ArrayBuffer {
  // Build header and calculate offsets
  const header: Record<string, { dtype: string; shape: number[]; data_offsets: [number, number] }> = {};
  let offset = 0;

  const dataBuffers: ArrayBuffer[] = [];

  for (const [name, tensor] of Object.entries(tensors)) {
    const dataLen = tensor.data.byteLength;
    header[name] = {
      dtype: tensor.dtype,
      shape: tensor.shape,
      data_offsets: [offset, offset + dataLen],
    };
    offset += dataLen;
    dataBuffers.push(tensor.data);
  }

  const headerJson = JSON.stringify(header);
  const headerBytes = new TextEncoder().encode(headerJson);

  // Total size: 8 (header size) + header + data
  const totalDataSize = dataBuffers.reduce((acc, b) => acc + b.byteLength, 0);
  const totalSize = 8 + headerBytes.length + totalDataSize;
  const buffer = new ArrayBuffer(totalSize);
  const view = new DataView(buffer);

  // Write header size (little-endian u64, using u32 low word)
  view.setUint32(0, headerBytes.length, true);
  view.setUint32(4, 0, true);

  // Write header
  new Uint8Array(buffer, 8, headerBytes.length).set(headerBytes);

  // Write tensor data
  let dataOffset = 8 + headerBytes.length;
  for (const dataBuffer of dataBuffers) {
    new Uint8Array(buffer, dataOffset, dataBuffer.byteLength).set(new Uint8Array(dataBuffer));
    dataOffset += dataBuffer.byteLength;
  }

  return buffer;
}

describe("SafeTensorsLoader", () => {
  it("should parse F32 tensor correctly", () => {
    const tensorData = new Float32Array([1, 2, 3, 4, 5, 6]);

    const buffer = createSafeTensorsBuffer({
      test_tensor: {
        dtype: "F32",
        shape: [2, 3],
        data: tensorData.buffer,
      },
    });

    const loader = SafeTensorsLoader.fromBuffer(buffer);
    const tensor = loader.getTensorFloat32("test_tensor");

    expect(tensor.shape).toEqual([2, 3]);
    expect(tensor.data).toHaveLength(6);
    expect(tensor.data[0]).toBe(1);
    expect(tensor.data[5]).toBe(6);
  });

  it("should list tensor names", () => {
    const buffer = createSafeTensorsBuffer({
      tensor_a: { dtype: "F32", shape: [1], data: new Float32Array([1]).buffer },
      tensor_b: { dtype: "F32", shape: [1], data: new Float32Array([2]).buffer },
    });

    const loader = SafeTensorsLoader.fromBuffer(buffer);
    const names = loader.getTensorNames();

    expect(names).toContain("tensor_a");
    expect(names).toContain("tensor_b");
    expect(names).toHaveLength(2);
  });

  it("should check tensor existence", () => {
    const buffer = createSafeTensorsBuffer({
      existing: { dtype: "F32", shape: [1], data: new Float32Array([42]).buffer },
    });

    const loader = SafeTensorsLoader.fromBuffer(buffer);

    expect(loader.hasTensor("existing")).toBe(true);
    expect(loader.hasTensor("nonexistent")).toBe(false);
  });

  it("should throw for missing tensor", () => {
    const buffer = createSafeTensorsBuffer({
      existing: { dtype: "F32", shape: [1], data: new Float32Array([42]).buffer },
    });

    const loader = SafeTensorsLoader.fromBuffer(buffer);

    expect(() => loader.getTensorFloat32("nonexistent")).toThrow("Tensor not found");
  });

  it("should load U8 tensor for packed weights", () => {
    const tensorData = new Uint8Array([0x05, 0x0A, 0x0F, 0x14]);

    const buffer = createSafeTensorsBuffer({
      packed_weights: {
        dtype: "U8",
        shape: [2, 2],
        data: tensorData.buffer,
      },
    });

    const loader = SafeTensorsLoader.fromBuffer(buffer);
    const tensor = loader.getTensorUint8("packed_weights");

    expect(tensor.shape).toEqual([2, 2]);
    expect(tensor.data).toHaveLength(4);
    expect(tensor.data[0]).toBe(0x05);
    expect(tensor.data[3]).toBe(0x14);
  });

  it("should handle multiple tensors", () => {
    const t1 = new Float32Array([1, 2, 3]);
    const t2 = new Float32Array([4, 5]);
    const t3 = new Uint8Array([6, 7, 8, 9]);

    const buffer = createSafeTensorsBuffer({
      tensor1: { dtype: "F32", shape: [3], data: t1.buffer },
      tensor2: { dtype: "F32", shape: [2], data: t2.buffer },
      tensor3: { dtype: "U8", shape: [4], data: t3.buffer },
    });

    const loader = SafeTensorsLoader.fromBuffer(buffer);

    const result1 = loader.getTensorFloat32("tensor1");
    expect(result1.data[0]).toBe(1);
    expect(result1.data[2]).toBe(3);

    const result2 = loader.getTensorFloat32("tensor2");
    expect(result2.data[0]).toBe(4);
    expect(result2.data[1]).toBe(5);

    const result3 = loader.getTensorUint8("tensor3");
    expect(result3.data[0]).toBe(6);
    expect(result3.data[3]).toBe(9);
  });

  it("should report correct byte length", () => {
    const buffer = createSafeTensorsBuffer({
      tensor: { dtype: "F32", shape: [4], data: new Float32Array([1, 2, 3, 4]).buffer },
    });

    const loader = SafeTensorsLoader.fromBuffer(buffer);

    expect(loader.byteLength).toBe(buffer.byteLength);
  });
});
