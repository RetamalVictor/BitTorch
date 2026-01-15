/**
 * BPE Tokenizer for Browser
 *
 * Implements Byte-Pair Encoding compatible with HuggingFace tokenizers.
 *
 * @module tokenizers/BPETokenizer
 */
/** Configuration for BPE tokenizer */
export interface TokenizerConfig {
    vocab: Map<string, number>;
    merges: Array<[string, string]>;
    decoder: Map<number, string>;
    unkToken: string;
    unkId: number;
}
/**
 * Byte-Pair Encoding tokenizer.
 *
 * Compatible with HuggingFace tokenizer.json format.
 *
 * @example
 * ```typescript
 * const tokenizer = await BPETokenizer.fromUrl('/model/tokenizer.json');
 *
 * // Encode text to token IDs
 * const ids = tokenizer.encode('Hello, world!');
 *
 * // Decode token IDs back to text
 * const text = tokenizer.decode(ids);
 * ```
 */
export declare class BPETokenizer {
    private vocab;
    private merges;
    private mergeRanks;
    private decoder;
    private unkId;
    private constructor();
    /**
     * Load tokenizer from HuggingFace JSON format.
     *
     * @param url - URL to tokenizer.json file
     * @returns Promise resolving to BPETokenizer instance
     */
    static fromUrl(url: string): Promise<BPETokenizer>;
    /**
     * Create tokenizer from parsed JSON.
     *
     * @param data - Parsed tokenizer.json contents
     * @returns BPETokenizer instance
     */
    static fromJSON(data: Record<string, unknown>): BPETokenizer;
    /**
     * Get vocabulary size.
     */
    get vocabSize(): number;
    /**
     * Encode text to token IDs.
     *
     * @param text - Input text to tokenize
     * @returns Array of token IDs
     */
    encode(text: string): number[];
    /**
     * Pre-tokenize: split on whitespace while preserving it as prefix.
     */
    private preTokenize;
    /**
     * Encode a single word using BPE.
     */
    private encodeWord;
    /**
     * Decode token IDs to text.
     *
     * @param ids - Array of token IDs
     * @returns Decoded text string
     */
    decode(ids: number[]): string;
    /**
     * Decode a single token ID to string.
     *
     * @param id - Token ID
     * @returns Token string
     */
    decodeToken(id: number): string;
    /**
     * Get token ID for a single token string.
     *
     * @param token - Token string
     * @returns Token ID or unknown ID if not found
     */
    tokenToId(token: string): number;
    /**
     * Get token string for an ID.
     *
     * @param id - Token ID
     * @returns Token string or empty string if not found
     */
    idToToken(id: number): string;
}
//# sourceMappingURL=BPETokenizer.d.ts.map