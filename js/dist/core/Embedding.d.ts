/**
 * Embedding - Token embedding lookup
 *
 * @module core/Embedding
 */
/**
 * Look up embeddings for a sequence of tokens.
 *
 * @param tokenIds - Token IDs [seqLen]
 * @param embedding - Embedding matrix [vocabSize, dim]
 * @param dim - Embedding dimension
 * @returns Embedded vectors [seqLen, dim]
 */
export declare function embeddingLookup(tokenIds: number[], embedding: Float32Array, dim: number): Float32Array;
/**
 * Look up embedding for a single token.
 *
 * @param tokenId - Token ID
 * @param embedding - Embedding matrix [vocabSize, dim]
 * @param dim - Embedding dimension
 * @returns Embedded vector [dim]
 */
export declare function embeddingLookupSingle(tokenId: number, embedding: Float32Array, dim: number): Float32Array;
//# sourceMappingURL=Embedding.d.ts.map