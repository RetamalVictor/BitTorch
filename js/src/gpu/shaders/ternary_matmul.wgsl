/**
 * Ternary Matrix Multiplication Compute Shader
 *
 * Performs Y = X * W^T * scales where W is ternary-quantized (2 bits per weight).
 *
 * Weight encoding (2 bits each, 4 weights per byte):
 *   0b00 = 0
 *   0b01 = +1
 *   0b10 = -1
 *
 * Memory layout:
 *   - input: [M, K] row-major (M = seqLen, K = inFeatures)
 *   - weights_packed: [N, K_bytes] where K_bytes is ALIGNED to 4 bytes, stored as u32 array
 *   - scales: [N] per-output-channel scales
 *   - output: [M, N] row-major (N = outFeatures)
 *
 * IMPORTANT: K_bytes must be a multiple of 4 for correct u32 indexing.
 * The host code pads weights if necessary.
 */

struct Uniforms {
    M: u32,         // Number of input rows (seqLen)
    N: u32,         // Number of output features
    K: u32,         // Number of input features (actual, not padded)
    K_bytes: u32,   // Aligned to multiple of 4: ceil(ceil(K/4) / 4) * 4
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights_packed: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> uniforms: Uniforms;

// Decode 2-bit ternary code to float
fn decode_ternary(code: u32) -> f32 {
    // 0 = 0, 1 = +1, 2 = -1
    if (code == 1u) { return 1.0; }
    if (code == 2u) { return -1.0; }
    return 0.0;
}

@compute @workgroup_size(64, 4, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;  // Input token index (M dimension)
    let col = gid.x;  // Output neuron index (N dimension)

    // Bounds check
    if (row >= uniforms.M || col >= uniforms.N) {
        return;
    }

    var acc: f32 = 0.0;
    let input_offset = row * uniforms.K;
    let weight_row_offset = col * uniforms.K_bytes;

    // K_bytes is aligned to 4, so we can index directly as u32 array
    // Each u32 = 4 bytes = 16 ternary weights (4 weights/byte * 4 bytes)
    let num_u32s = uniforms.K_bytes / 4u;

    for (var u32_idx: u32 = 0u; u32_idx < num_u32s; u32_idx++) {
        let packed_u32 = weights_packed[weight_row_offset / 4u + u32_idx];

        // Each u32 contains 4 bytes, each byte contains 4 ternary weights
        for (var byte_in_u32: u32 = 0u; byte_in_u32 < 4u; byte_in_u32++) {
            let byte_idx = u32_idx * 4u + byte_in_u32;
            if (byte_idx >= uniforms.K_bytes) { break; }

            // Extract this byte from the u32
            let byte_val = (packed_u32 >> (byte_in_u32 * 8u)) & 0xFFu;

            // Unpack 4 ternary values from this byte
            for (var i: u32 = 0u; i < 4u; i++) {
                let k = byte_idx * 4u + i;
                if (k >= uniforms.K) { break; }

                let code = (byte_val >> (i * 2u)) & 0x3u;
                let w = decode_ternary(code);
                acc += input[input_offset + k] * w;
            }
        }
    }

    // Apply per-channel scale
    output[row * uniforms.N + col] = acc * scales[col];
}
