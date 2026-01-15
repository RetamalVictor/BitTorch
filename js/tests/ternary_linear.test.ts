/**
 * Tests for ternary linear operations
 */

import { describe, it, expect } from "vitest";
import {
  ternaryMatmul,
  ternaryMatmulSingle,
  matmulFP32,
} from "../src/core/TernaryLinear.js";
import type { TernaryLayer } from "../src/core/types.js";

/**
 * Pack ternary weights into 2-bit format.
 * Encoding: 0b00 = 0, 0b01 = +1, 0b10 = -1
 */
function packTernary(weights: number[][], outFeatures: number, inFeatures: number): Uint8Array {
  const kBytes = Math.ceil(inFeatures / 4);
  const packed = new Uint8Array(outFeatures * kBytes);

  for (let n = 0; n < outFeatures; n++) {
    for (let k = 0; k < inFeatures; k++) {
      const byteIdx = Math.floor(k / 4);
      const bitPos = (k % 4) * 2;
      const w = weights[n][k];
      let code = 0;
      if (w === 1) code = 1;
      else if (w === -1) code = 2;
      packed[n * kBytes + byteIdx] |= code << bitPos;
    }
  }

  return packed;
}

describe("ternaryMatmul", () => {
  it("should compute matmul with identity-like ternary weights", () => {
    // Create a simple ternary layer: 4 -> 4, identity-ish
    const weights = [
      [1, 0, 0, 0],  // row 0 picks input[0]
      [0, 1, 0, 0],  // row 1 picks input[1]
      [0, 0, 1, 0],  // row 2 picks input[2]
      [0, 0, 0, 1],  // row 3 picks input[3]
    ];

    const layer: TernaryLayer = {
      weightsPacked: packTernary(weights, 4, 4),
      scales: new Float32Array([1, 1, 1, 1]),
      outFeatures: 4,
      inFeatures: 4,
    };

    const input = new Float32Array([1, 2, 3, 4]);
    const output = ternaryMatmulSingle(input, layer);

    expect(output).toHaveLength(4);
    expect(output[0]).toBeCloseTo(1);
    expect(output[1]).toBeCloseTo(2);
    expect(output[2]).toBeCloseTo(3);
    expect(output[3]).toBeCloseTo(4);
  });

  it("should apply per-channel scales", () => {
    const weights = [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
    ];

    const layer: TernaryLayer = {
      weightsPacked: packTernary(weights, 2, 4),
      scales: new Float32Array([2.0, 0.5]),
      outFeatures: 2,
      inFeatures: 4,
    };

    const input = new Float32Array([4, 8, 0, 0]);
    const output = ternaryMatmulSingle(input, layer);

    expect(output).toHaveLength(2);
    expect(output[0]).toBeCloseTo(8);  // 4 * 1 * 2.0
    expect(output[1]).toBeCloseTo(4);  // 8 * 1 * 0.5
  });

  it("should handle negative weights", () => {
    const weights = [
      [1, -1, 0, 0],  // computes input[0] - input[1]
      [-1, 1, 0, 0],  // computes -input[0] + input[1]
    ];

    const layer: TernaryLayer = {
      weightsPacked: packTernary(weights, 2, 4),
      scales: new Float32Array([1, 1]),
      outFeatures: 2,
      inFeatures: 4,
    };

    const input = new Float32Array([10, 3, 0, 0]);
    const output = ternaryMatmulSingle(input, layer);

    expect(output[0]).toBeCloseTo(7);   // 10 - 3
    expect(output[1]).toBeCloseTo(-7);  // -10 + 3
  });

  it("should handle batch inputs", () => {
    const weights = [
      [1, 1],
      [-1, 1],
    ];

    const layer: TernaryLayer = {
      weightsPacked: packTernary(weights, 2, 2),
      scales: new Float32Array([1, 1]),
      outFeatures: 2,
      inFeatures: 2,
    };

    // 2 sequences, 2 features each
    const input = new Float32Array([
      1, 2,  // seq 0
      3, 4,  // seq 1
    ]);

    const output = ternaryMatmul(input, layer, 2);

    expect(output).toHaveLength(4);
    expect(output[0]).toBeCloseTo(3);   // 1 + 2
    expect(output[1]).toBeCloseTo(1);   // -1 + 2
    expect(output[2]).toBeCloseTo(7);   // 3 + 4
    expect(output[3]).toBeCloseTo(1);   // -3 + 4
  });
});

describe("matmulFP32", () => {
  it("should compute standard FP32 matmul", () => {
    // 1x2 @ 2x3 -> 1x3
    const input = new Float32Array([1, 2]);
    const weights = new Float32Array([
      // column-major: each column is one output
      1, 3,   // column 0
      2, 4,   // column 1
      5, 6,   // column 2
    ]);

    const output = matmulFP32(input, weights, 2, 3);

    // output[0] = 1*1 + 2*3 = 7
    // output[1] = 1*2 + 2*4 = 10
    // output[2] = 1*5 + 2*6 = 17
    expect(output).toHaveLength(3);
    expect(output[0]).toBeCloseTo(7);
    expect(output[1]).toBeCloseTo(10);
    expect(output[2]).toBeCloseTo(17);
  });
});
