/**
 * WebGPU Backend for Ternary Matrix Multiplication
 *
 * Provides GPU-accelerated ternary matmul for transformer inference.
 *
 * @module gpu/WebGPUBackend
 */
import { TERNARY_MATMUL_SHADER } from "./shaders/index.js";
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
export class WebGPUBackend {
    device;
    pipeline;
    bindGroupLayout;
    // Pre-uploaded weight buffers per layer
    layerBuffers = new Map();
    // Reusable buffers (sized for largest layer)
    inputBuffer = null;
    outputBuffer = null;
    stagingBuffer = null;
    uniformBuffer;
    // Track max sizes for buffer allocation
    maxInputSize = 0;
    maxOutputSize = 0;
    constructor(device, pipeline, bindGroupLayout) {
        this.device = device;
        this.pipeline = pipeline;
        this.bindGroupLayout = bindGroupLayout;
        // Create uniform buffer (16 bytes for 4 u32s)
        this.uniformBuffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }
    /**
     * Create a WebGPU backend instance.
     *
     * @returns Backend instance or null if WebGPU is unavailable
     */
    static async create() {
        if (typeof navigator === "undefined" || !navigator.gpu) {
            return null;
        }
        try {
            const adapter = await navigator.gpu.requestAdapter({
                powerPreference: "high-performance",
            });
            if (!adapter) {
                return null;
            }
            // Request device with limits capped to what adapter supports
            const limits = adapter.limits;
            const desiredStorageSize = 128 * 1024 * 1024; // 128MB
            const desiredBufferSize = 256 * 1024 * 1024; // 256MB
            const device = await adapter.requestDevice({
                requiredLimits: {
                    maxStorageBufferBindingSize: Math.min(desiredStorageSize, limits.maxStorageBufferBindingSize),
                    maxBufferSize: Math.min(desiredBufferSize, limits.maxBufferSize),
                },
            });
            // Handle device loss
            device.lost.then((info) => {
                console.error("[WebGPUBackend] Device lost:", info.message);
            });
            // Compile shader
            const shaderModule = device.createShaderModule({
                code: TERNARY_MATMUL_SHADER,
            });
            // Check for compilation errors
            const compilationInfo = await shaderModule.getCompilationInfo();
            for (const message of compilationInfo.messages) {
                if (message.type === "error") {
                    console.error("[WebGPUBackend] Shader compilation failed");
                    return null;
                }
            }
            // Create bind group layout
            const bindGroupLayout = device.createBindGroupLayout({
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: "read-only-storage" },
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: "read-only-storage" },
                    },
                    {
                        binding: 2,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: "read-only-storage" },
                    },
                    {
                        binding: 3,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: "storage" },
                    },
                    {
                        binding: 4,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: "uniform" },
                    },
                ],
            });
            // Create pipeline
            const pipeline = device.createComputePipeline({
                layout: device.createPipelineLayout({
                    bindGroupLayouts: [bindGroupLayout],
                }),
                compute: {
                    module: shaderModule,
                    entryPoint: "main",
                },
            });
            return new WebGPUBackend(device, pipeline, bindGroupLayout);
        }
        catch (error) {
            console.error("[WebGPUBackend] Failed to initialize:", error);
            return null;
        }
    }
    /**
     * Upload a layer's weights to GPU.
     *
     * Call once per layer at initialization time.
     *
     * @param name - Unique layer identifier
     * @param layer - Ternary layer with packed weights and scales
     */
    uploadLayer(name, layer) {
        const { weightsPacked, scales, outFeatures, inFeatures } = layer;
        const kBytes = Math.ceil(inFeatures / 4);
        // Align kBytes to multiple of 4 for u32 access in shader
        const kBytesAligned = Math.ceil(kBytes / 4) * 4;
        let weightsData;
        if (kBytes === kBytesAligned) {
            weightsData = weightsPacked;
        }
        else {
            // Repack weights with row padding for alignment
            weightsData = new Uint8Array(outFeatures * kBytesAligned);
            for (let row = 0; row < outFeatures; row++) {
                const srcOffset = row * kBytes;
                const dstOffset = row * kBytesAligned;
                weightsData.set(weightsPacked.subarray(srcOffset, srcOffset + kBytes), dstOffset);
            }
        }
        // Create weight buffer
        const weightsBuffer = this.device.createBuffer({
            size: weightsData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(weightsBuffer, 0, weightsData);
        // Create scales buffer
        const scalesBuffer = this.device.createBuffer({
            size: scales.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(scalesBuffer, 0, scales);
        this.layerBuffers.set(name, {
            weightsPacked: weightsBuffer,
            scales: scalesBuffer,
            outFeatures,
            inFeatures,
            kBytes: kBytesAligned,
        });
        // Track max sizes for dynamic buffer allocation
        this.maxInputSize = Math.max(this.maxInputSize, inFeatures);
        this.maxOutputSize = Math.max(this.maxOutputSize, outFeatures);
    }
    /**
     * Ensure activation buffers are large enough.
     */
    ensureBuffers(seqLen) {
        const requiredInputSize = seqLen * this.maxInputSize * 4;
        const requiredOutputSize = seqLen * this.maxOutputSize * 4;
        if (!this.inputBuffer || this.inputBuffer.size < requiredInputSize) {
            this.inputBuffer?.destroy();
            this.inputBuffer = this.device.createBuffer({
                size: requiredInputSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
        }
        if (!this.outputBuffer || this.outputBuffer.size < requiredOutputSize) {
            this.outputBuffer?.destroy();
            this.outputBuffer = this.device.createBuffer({
                size: requiredOutputSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            });
        }
        if (!this.stagingBuffer || this.stagingBuffer.size < requiredOutputSize) {
            this.stagingBuffer?.destroy();
            this.stagingBuffer = this.device.createBuffer({
                size: requiredOutputSize,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });
        }
    }
    /**
     * Execute a ternary matrix multiplication on GPU.
     *
     * @param input - Input tensor [seqLen, inFeatures]
     * @param layerName - Name of pre-uploaded layer
     * @param seqLen - Sequence length (number of input rows)
     * @returns Output tensor [seqLen, outFeatures]
     */
    async matmul(input, layerName, seqLen) {
        const layer = this.layerBuffers.get(layerName);
        if (!layer) {
            throw new Error(`[WebGPUBackend] Layer not found: ${layerName}`);
        }
        this.ensureBuffers(seqLen);
        // Upload input
        this.device.queue.writeBuffer(this.inputBuffer, 0, input);
        // Update uniforms
        const uniforms = new Uint32Array([
            seqLen,
            layer.outFeatures,
            layer.inFeatures,
            layer.kBytes,
        ]);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniforms);
        // Create bind group
        const bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.inputBuffer } },
                { binding: 1, resource: { buffer: layer.weightsPacked } },
                { binding: 2, resource: { buffer: layer.scales } },
                { binding: 3, resource: { buffer: this.outputBuffer } },
                { binding: 4, resource: { buffer: this.uniformBuffer } },
            ],
        });
        // Dispatch
        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, bindGroup);
        // Workgroup size is 64x4, so dispatch ceil(N/64) x ceil(M/4)
        const workgroupsX = Math.ceil(layer.outFeatures / 64);
        const workgroupsY = Math.ceil(seqLen / 4);
        pass.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
        pass.end();
        // Copy to staging buffer
        const outputSize = seqLen * layer.outFeatures * 4;
        encoder.copyBufferToBuffer(this.outputBuffer, 0, this.stagingBuffer, 0, outputSize);
        // Submit and wait
        this.device.queue.submit([encoder.finish()]);
        // Read back results
        await this.stagingBuffer.mapAsync(GPUMapMode.READ);
        const resultData = new Float32Array(this.stagingBuffer.getMappedRange(0, outputSize).slice(0));
        this.stagingBuffer.unmap();
        return resultData;
    }
    /**
     * Check if a layer has been uploaded.
     */
    hasLayer(name) {
        return this.layerBuffers.has(name);
    }
    /**
     * Get memory usage statistics.
     */
    getMemoryStats() {
        let layerTotal = 0;
        for (const layer of this.layerBuffers.values()) {
            layerTotal += layer.weightsPacked.size + layer.scales.size;
        }
        const activationTotal = (this.inputBuffer?.size ?? 0) +
            (this.outputBuffer?.size ?? 0) +
            (this.stagingBuffer?.size ?? 0);
        return {
            layerBuffersKB: Math.round(layerTotal / 1024),
            activationBuffersKB: Math.round(activationTotal / 1024),
        };
    }
    /**
     * Release all GPU resources.
     */
    destroy() {
        for (const layer of this.layerBuffers.values()) {
            layer.weightsPacked.destroy();
            layer.scales.destroy();
        }
        this.layerBuffers.clear();
        this.inputBuffer?.destroy();
        this.outputBuffer?.destroy();
        this.stagingBuffer?.destroy();
        this.uniformBuffer.destroy();
        this.device.destroy();
    }
}
//# sourceMappingURL=WebGPUBackend.js.map