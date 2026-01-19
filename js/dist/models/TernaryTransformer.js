/**
 * TernaryTransformer - LLaMA-style transformer with ternary quantization
 *
 * Main model class for browser-based inference with ternary quantized weights.
 *
 * @module models/TernaryTransformer
 */
import { ternaryMatmul, ternaryMatmulSingle, matmulFP32 } from "../core/TernaryLinear.js";
import { rmsNorm, rmsNormSingle } from "../core/RMSNorm.js";
import { embeddingLookup, embeddingLookupSingle } from "../core/Embedding.js";
import { initRoPECache, applyRoPE, applyRoPESingle, applyRoPEToCache, applyRoPEToSinglePos, } from "../core/RoPE.js";
import { SafeTensorsLoader } from "../loaders/SafeTensorsLoader.js";
import { BPETokenizer } from "../tokenizers/BPETokenizer.js";
import { WebGPUBackend } from "../gpu/WebGPUBackend.js";
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
export class TernaryTransformer {
    _config;
    tokenizer;
    // Model weights
    embedding;
    blocks = [];
    normWeight;
    head;
    // RoPE cache
    ropeCache;
    // KV Cache
    kvCache = [];
    cacheSeqLen = 0;
    // GPU acceleration
    gpu = null;
    // Generation control
    stopRequested = false;
    // Memory tracking
    ternaryWeights = 0;
    fp16Weights = 0;
    constructor(config, tokenizer, embedding, blocks, normWeight, head, ropeCache, gpu) {
        this._config = config;
        this.tokenizer = tokenizer;
        this.embedding = embedding;
        this.blocks = blocks;
        this.normWeight = normWeight;
        this.head = head;
        this.ropeCache = ropeCache;
        this.gpu = gpu;
    }
    /**
     * Load model from a directory containing config.json, tokenizer.json, and model.safetensors.
     *
     * @param basePath - URL or path to model directory
     * @returns Promise resolving to TernaryTransformer instance
     */
    static async load(basePath) {
        // Load config
        const configResp = await fetch(`${basePath}/config.json`);
        if (!configResp.ok)
            throw new Error("Failed to load config.json");
        const configJson = await configResp.json();
        const config = {
            vocabSize: configJson.vocab_size,
            dim: configJson.dim,
            nLayers: configJson.n_layers,
            nHeads: configJson.n_heads,
            nKvHeads: configJson.n_kv_heads ?? configJson.n_heads,
            maxSeqLen: configJson.max_seq_len,
        };
        // Load tokenizer
        const tokenizer = await BPETokenizer.fromUrl(`${basePath}/tokenizer.json`);
        // Load SafeTensors
        const loader = await SafeTensorsLoader.fromUrl(`${basePath}/model.safetensors`);
        // Load embedding
        const tokTensor = loader.getTensorFloat32("tok.weight");
        const embedding = tokTensor.data;
        // Load head
        const headTensor = loader.getTensorFloat32("head.weight");
        const head = headTensor.data;
        // Load final norm
        const normTensor = loader.getTensorFloat32("norm.weight");
        const normWeight = normTensor.data;
        // Load transformer blocks
        const blocks = [];
        let ternaryWeights = 0;
        const fp16Weights = config.vocabSize * config.dim * 2; // embedding + head
        for (let i = 0; i < config.nLayers; i++) {
            const prefix = `blocks.${i}`;
            // RMSNorm weights
            const norm1 = loader.getTensorFloat32(`${prefix}.norm1.weight`);
            const norm2 = loader.getTensorFloat32(`${prefix}.norm2.weight`);
            // Helper to load ternary layer
            // Model stores weights as [inBytes, outFeatures] but matmul expects [outFeatures, inBytes]
            // So we transpose during loading
            const loadTernary = (name) => {
                const weightTensor = loader.getTensorUint8(`${name}.weight_packed`);
                const scaleTensor = loader.getTensorFloat32(`${name}.scale`);
                // Scale length gives us outFeatures (one scale per output channel)
                const outFeatures = scaleTensor.data.length;
                // shape[0] is inBytes (packed), so inFeatures = shape[0] * 4
                const inBytes = weightTensor.shape[0];
                const inFeatures = inBytes * 4;
                // Transpose weights from [inBytes, outFeatures] to [outFeatures, inBytes]
                const transposedWeights = new Uint8Array(outFeatures * inBytes);
                const srcWeights = weightTensor.data;
                for (let n = 0; n < outFeatures; n++) {
                    for (let b = 0; b < inBytes; b++) {
                        transposedWeights[n * inBytes + b] = srcWeights[b * outFeatures + n];
                    }
                }
                ternaryWeights += outFeatures * inFeatures;
                return {
                    weightsPacked: transposedWeights,
                    scales: scaleTensor.data,
                    outFeatures,
                    inFeatures,
                };
            };
            blocks.push({
                norm1Weight: norm1.data,
                qProj: loadTernary(`${prefix}.attn.q_proj`),
                kvProj: loadTernary(`${prefix}.attn.kv_proj`),
                proj: loadTernary(`${prefix}.attn.proj`),
                norm2Weight: norm2.data,
                wGate: loadTernary(`${prefix}.mlp.w_gate`),
                wUp: loadTernary(`${prefix}.mlp.w_up`),
                wDown: loadTernary(`${prefix}.mlp.w_down`),
            });
        }
        // Initialize RoPE cache
        const headDim = config.dim / config.nHeads;
        const ropeCache = initRoPECache(config.maxSeqLen, headDim);
        // Try to initialize GPU
        let gpu = null;
        try {
            gpu = await WebGPUBackend.create();
            if (gpu) {
                // Upload all weights to GPU
                for (let i = 0; i < blocks.length; i++) {
                    const block = blocks[i];
                    const prefix = `block${i}`;
                    gpu.uploadLayer(`${prefix}_qProj`, block.qProj);
                    gpu.uploadLayer(`${prefix}_kvProj`, block.kvProj);
                    gpu.uploadLayer(`${prefix}_proj`, block.proj);
                    gpu.uploadLayer(`${prefix}_wGate`, block.wGate);
                    gpu.uploadLayer(`${prefix}_wUp`, block.wUp);
                    gpu.uploadLayer(`${prefix}_wDown`, block.wDown);
                }
            }
        }
        catch {
            gpu = null;
        }
        const model = new TernaryTransformer(config, tokenizer, embedding, blocks, normWeight, head, ropeCache, gpu);
        model.ternaryWeights = ternaryWeights;
        model.fp16Weights = fp16Weights;
        return model;
    }
    /**
     * Get model configuration.
     */
    get config() {
        return { ...this._config };
    }
    /**
     * Check if GPU acceleration is enabled.
     */
    get isGPUEnabled() {
        return this.gpu !== null;
    }
    /**
     * Get memory usage statistics.
     */
    get memoryStats() {
        const packedWeightsKB = (this.ternaryWeights * 0.25) / 1024;
        const ternaryAsFP16KB = (this.ternaryWeights * 2) / 1024;
        const actualFP16KB = (this.fp16Weights * 2) / 1024;
        const numScales = this.blocks.reduce((acc, b) => acc +
            b.qProj.outFeatures +
            b.kvProj.outFeatures +
            b.proj.outFeatures +
            b.wGate.outFeatures +
            b.wUp.outFeatures +
            b.wDown.outFeatures, 0);
        const scalesKB = (numScales * 4) / 1024;
        return {
            packedWeightsKB: packedWeightsKB + scalesKB,
            fp16EquivalentKB: ternaryAsFP16KB + actualFP16KB,
            compressionRatio: (ternaryAsFP16KB + actualFP16KB) /
                (packedWeightsKB + scalesKB + actualFP16KB),
            scalesKB,
        };
    }
    /**
     * Generate text continuation.
     *
     * @param prompt - Starting text
     * @param options - Generation options
     * @returns Generated text
     */
    async generate(prompt, options = {}) {
        const { maxTokens = 100, temperature = 0.8, onToken } = options;
        this.stopRequested = false;
        const startTime = performance.now();
        // Tokenize prompt
        const tokens = this.tokenizer.encode(prompt);
        // Initialize KV cache
        this.initKVCache();
        // Prefill: process all prompt tokens
        let logits = await this.forwardPrefill(tokens);
        let generated = "";
        for (let step = 0; step < maxTokens; step++) {
            if (this.stopRequested)
                break;
            // Sample next token
            const nextToken = this.sample(logits, temperature);
            // Decode
            const decoded = this.tokenizer.decodeToken(nextToken);
            generated += decoded;
            // Report progress
            if (onToken) {
                const elapsedMs = performance.now() - startTime;
                onToken(decoded, {
                    tokensPerSecond: ((step + 1) / elapsedMs) * 1000,
                    totalTokens: step + 1,
                    elapsedMs,
                });
            }
            // Yield to UI
            await new Promise((resolve) => setTimeout(resolve, 0));
            // Check max sequence length
            if (this.cacheSeqLen >= this._config.maxSeqLen - 1)
                break;
            // Decode step
            logits = await this.forwardDecode(nextToken);
        }
        return generated;
    }
    /**
     * Stop ongoing generation.
     */
    stop() {
        this.stopRequested = true;
    }
    /**
     * Release all resources.
     */
    destroy() {
        this.gpu?.destroy();
        this.gpu = null;
    }
    // ============================================
    // Private Methods
    // ============================================
    initKVCache() {
        const { nLayers, nKvHeads, maxSeqLen, dim, nHeads } = this._config;
        const headDim = dim / nHeads;
        this.kvCache = [];
        for (let i = 0; i < nLayers; i++) {
            this.kvCache.push({
                k: new Float32Array(nKvHeads * maxSeqLen * headDim),
                v: new Float32Array(nKvHeads * maxSeqLen * headDim),
            });
        }
        this.cacheSeqLen = 0;
    }
    async forwardPrefill(tokens) {
        const { dim, nHeads, nLayers } = this._config;
        const seqLen = tokens.length;
        const headDim = dim / nHeads;
        // Embedding lookup
        let hidden = embeddingLookup(tokens, this.embedding, dim);
        // Process blocks
        for (let i = 0; i < nLayers; i++) {
            const block = this.blocks[i];
            const cache = this.kvCache[i];
            // Pre-attention RMSNorm
            const normed1 = rmsNorm(hidden, block.norm1Weight, seqLen, dim);
            // Attention
            const attnOut = await this.attentionPrefill(normed1, block, cache, seqLen, headDim, i);
            // Residual
            for (let j = 0; j < hidden.length; j++) {
                hidden[j] += attnOut[j];
            }
            // Pre-MLP RMSNorm
            const normed2 = rmsNorm(hidden, block.norm2Weight, seqLen, dim);
            // MLP (SwiGLU)
            const mlpOut = await this.swiglu(normed2, block, seqLen, i);
            // Residual
            for (let j = 0; j < hidden.length; j++) {
                hidden[j] += mlpOut[j];
            }
        }
        this.cacheSeqLen = seqLen;
        // Final RMSNorm
        const finalHidden = rmsNorm(hidden, this.normWeight, seqLen, dim);
        // Get last token's hidden state
        const lastHidden = new Float32Array(dim);
        const lastOffset = (seqLen - 1) * dim;
        for (let d = 0; d < dim; d++) {
            lastHidden[d] = finalHidden[lastOffset + d];
        }
        // Output projection
        return matmulFP32(lastHidden, this.head, dim, this._config.vocabSize);
    }
    async forwardDecode(token) {
        const { dim, nHeads, nLayers } = this._config;
        const headDim = dim / nHeads;
        const pos = this.cacheSeqLen;
        // Embedding lookup
        let hidden = embeddingLookupSingle(token, this.embedding, dim);
        // Process blocks
        for (let i = 0; i < nLayers; i++) {
            const block = this.blocks[i];
            const cache = this.kvCache[i];
            const normed1 = rmsNormSingle(hidden, block.norm1Weight);
            const attnOut = await this.attentionDecode(normed1, block, cache, pos, headDim, i);
            for (let d = 0; d < dim; d++) {
                hidden[d] += attnOut[d];
            }
            const normed2 = rmsNormSingle(hidden, block.norm2Weight);
            const mlpOut = await this.swigluSingle(normed2, block, i);
            for (let d = 0; d < dim; d++) {
                hidden[d] += mlpOut[d];
            }
        }
        this.cacheSeqLen = pos + 1;
        const finalHidden = rmsNormSingle(hidden, this.normWeight);
        return matmulFP32(finalHidden, this.head, dim, this._config.vocabSize);
    }
    async ternaryMatmulDispatch(input, layer, seqLen, layerName) {
        if (this.gpu) {
            return this.gpu.matmul(input, layerName, seqLen);
        }
        return ternaryMatmul(input, layer, seqLen);
    }
    async ternaryMatmulSingleDispatch(input, layer, layerName) {
        if (this.gpu) {
            return this.gpu.matmul(input, layerName, 1);
        }
        return ternaryMatmulSingle(input, layer);
    }
    async swiglu(x, block, seqLen, blockIdx) {
        const prefix = `block${blockIdx}`;
        const gate = await this.ternaryMatmulDispatch(x, block.wGate, seqLen, `${prefix}_wGate`);
        const up = await this.ternaryMatmulDispatch(x, block.wUp, seqLen, `${prefix}_wUp`);
        for (let i = 0; i < gate.length; i++) {
            const g = gate[i];
            const silu = g / (1 + Math.exp(-g));
            gate[i] = silu * up[i];
        }
        return this.ternaryMatmulDispatch(gate, block.wDown, seqLen, `${prefix}_wDown`);
    }
    async swigluSingle(x, block, blockIdx) {
        const prefix = `block${blockIdx}`;
        const gate = await this.ternaryMatmulSingleDispatch(x, block.wGate, `${prefix}_wGate`);
        const up = await this.ternaryMatmulSingleDispatch(x, block.wUp, `${prefix}_wUp`);
        for (let i = 0; i < gate.length; i++) {
            const g = gate[i];
            const silu = g / (1 + Math.exp(-g));
            gate[i] = silu * up[i];
        }
        return this.ternaryMatmulSingleDispatch(gate, block.wDown, `${prefix}_wDown`);
    }
    async attentionPrefill(x, block, cache, seqLen, headDim, blockIdx) {
        const { dim, nHeads, nKvHeads, maxSeqLen } = this._config;
        const kvDim = nKvHeads * headDim;
        const prefix = `block${blockIdx}`;
        const q = await this.ternaryMatmulDispatch(x, block.qProj, seqLen, `${prefix}_qProj`);
        const kv = await this.ternaryMatmulDispatch(x, block.kvProj, seqLen, `${prefix}_kvProj`);
        // Store K, V in cache
        for (let t = 0; t < seqLen; t++) {
            const kvOffset = t * 2 * kvDim;
            for (let kh = 0; kh < nKvHeads; kh++) {
                const cacheOffset = kh * maxSeqLen * headDim + t * headDim;
                const kOffset = kvOffset + kh * headDim;
                const vOffset = kvOffset + kvDim + kh * headDim;
                for (let d = 0; d < headDim; d++) {
                    cache.k[cacheOffset + d] = kv[kOffset + d];
                    cache.v[cacheOffset + d] = kv[vOffset + d];
                }
            }
        }
        // Apply RoPE
        applyRoPE(q, this.ropeCache, seqLen, nHeads, dim);
        applyRoPEToCache(cache.k, this.ropeCache, seqLen, nKvHeads, maxSeqLen);
        // Compute attention
        const scale = 1.0 / Math.sqrt(headDim);
        const attnOut = new Float32Array(seqLen * dim);
        const headsPerKv = nHeads / nKvHeads;
        for (let h = 0; h < nHeads; h++) {
            const kvHead = Math.floor(h / headsPerKv);
            for (let tq = 0; tq < seqLen; tq++) {
                const scores = new Float32Array(seqLen);
                let maxScore = -Infinity;
                for (let tk = 0; tk <= tq; tk++) {
                    let score = 0;
                    const qOffset = tq * dim + h * headDim;
                    const kOffset = kvHead * maxSeqLen * headDim + tk * headDim;
                    for (let d = 0; d < headDim; d++) {
                        score += q[qOffset + d] * cache.k[kOffset + d];
                    }
                    score *= scale;
                    scores[tk] = score;
                    if (score > maxScore)
                        maxScore = score;
                }
                // Softmax
                let sumExp = 0;
                for (let tk = 0; tk <= tq; tk++) {
                    scores[tk] = Math.exp(scores[tk] - maxScore);
                    sumExp += scores[tk];
                }
                for (let tk = 0; tk <= tq; tk++) {
                    scores[tk] /= sumExp;
                }
                // Weighted sum
                const outOffset = tq * dim + h * headDim;
                for (let d = 0; d < headDim; d++) {
                    let acc = 0;
                    for (let tk = 0; tk <= tq; tk++) {
                        const vOffset = kvHead * maxSeqLen * headDim + tk * headDim;
                        acc += scores[tk] * cache.v[vOffset + d];
                    }
                    attnOut[outOffset + d] = acc;
                }
            }
        }
        return this.ternaryMatmulDispatch(attnOut, block.proj, seqLen, `${prefix}_proj`);
    }
    async attentionDecode(x, block, cache, pos, headDim, blockIdx) {
        const { dim, nHeads, nKvHeads, maxSeqLen } = this._config;
        const kvDim = nKvHeads * headDim;
        const prefix = `block${blockIdx}`;
        const q = await this.ternaryMatmulSingleDispatch(x, block.qProj, `${prefix}_qProj`);
        const kv = await this.ternaryMatmulSingleDispatch(x, block.kvProj, `${prefix}_kvProj`);
        // Store K, V at current position
        for (let kh = 0; kh < nKvHeads; kh++) {
            const cacheOffset = kh * maxSeqLen * headDim + pos * headDim;
            const kOffset = kh * headDim;
            const vOffset = kvDim + kh * headDim;
            for (let d = 0; d < headDim; d++) {
                cache.k[cacheOffset + d] = kv[kOffset + d];
                cache.v[cacheOffset + d] = kv[vOffset + d];
            }
        }
        // Apply RoPE
        applyRoPESingle(q, this.ropeCache, pos, nHeads);
        applyRoPEToSinglePos(cache.k, this.ropeCache, pos, nKvHeads, maxSeqLen);
        // Compute attention
        const scale = 1.0 / Math.sqrt(headDim);
        const attnOut = new Float32Array(dim);
        const headsPerKv = nHeads / nKvHeads;
        const seqLen = pos + 1;
        for (let h = 0; h < nHeads; h++) {
            const kvHead = Math.floor(h / headsPerKv);
            const scores = new Float32Array(seqLen);
            let maxScore = -Infinity;
            for (let tk = 0; tk < seqLen; tk++) {
                let score = 0;
                const qOffset = h * headDim;
                const kOffset = kvHead * maxSeqLen * headDim + tk * headDim;
                for (let d = 0; d < headDim; d++) {
                    score += q[qOffset + d] * cache.k[kOffset + d];
                }
                score *= scale;
                scores[tk] = score;
                if (score > maxScore)
                    maxScore = score;
            }
            let sumExp = 0;
            for (let tk = 0; tk < seqLen; tk++) {
                scores[tk] = Math.exp(scores[tk] - maxScore);
                sumExp += scores[tk];
            }
            for (let tk = 0; tk < seqLen; tk++) {
                scores[tk] /= sumExp;
            }
            const outOffset = h * headDim;
            for (let d = 0; d < headDim; d++) {
                let acc = 0;
                for (let tk = 0; tk < seqLen; tk++) {
                    const vOffset = kvHead * maxSeqLen * headDim + tk * headDim;
                    acc += scores[tk] * cache.v[vOffset + d];
                }
                attnOut[outOffset + d] = acc;
            }
        }
        return this.ternaryMatmulSingleDispatch(attnOut, block.proj, `${prefix}_proj`);
    }
    sample(logits, temperature) {
        const scaled = new Float32Array(logits.length);
        let maxLogit = -Infinity;
        for (let i = 0; i < logits.length; i++) {
            scaled[i] = logits[i] / temperature;
            if (scaled[i] > maxLogit)
                maxLogit = scaled[i];
        }
        let sumExp = 0;
        for (let i = 0; i < scaled.length; i++) {
            scaled[i] = Math.exp(scaled[i] - maxLogit);
            sumExp += scaled[i];
        }
        for (let i = 0; i < scaled.length; i++) {
            scaled[i] /= sumExp;
        }
        const r = Math.random();
        let cumsum = 0;
        for (let i = 0; i < scaled.length; i++) {
            cumsum += scaled[i];
            if (r < cumsum)
                return i;
        }
        return scaled.length - 1;
    }
}
//# sourceMappingURL=TernaryTransformer.js.map