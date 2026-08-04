/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.highlight;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;

import org.apache.lucene.search.join.ScoreMode;
import org.elasticsearch.cluster.metadata.IndexMetadata;
import org.elasticsearch.common.bytes.BytesReference;
import org.elasticsearch.common.io.Streams;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.index.IndexVersion;
import org.elasticsearch.index.mapper.InferenceMetadataFieldsMapper;
import org.elasticsearch.index.mapper.SourceToParse;
import org.elasticsearch.index.query.NestedQueryBuilder;
import org.elasticsearch.inference.WeightedToken;
import org.elasticsearch.search.fetch.subphase.highlight.HighlightBuilder;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentFactory;
import org.elasticsearch.xcontent.XContentType;
import org.elasticsearch.xpack.core.ml.search.SparseVectorQueryBuilder;
import org.elasticsearch.xpack.inference.mapper.SemanticTextFieldMapper;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static org.elasticsearch.xpack.inference.mapper.SemanticInferenceMetadataFieldsMapperTests.getRandomCompatibleIndexVersion;

public class SemanticTextHighlighterTests extends AbstractSemanticHighlighterTests {
    private static final String SEMANTIC_FIELD_ELSER = "field-sparse-vector";

    /**
     * Sparse-vector query tokens. Each chunk document contains exactly one of these tokens
     * (stored with weight 1.0), so the score for a chunk equals the corresponding query weight.
     *
     * <p>Token-to-chunk assignment (designed so score order ≠ offset order):
     * <ul>
     *   <li>chunk 0 → tok_b (weight 0.80) → 2nd score</li>
     *   <li>chunk 1 → tok_d (weight 0.60) → 4th score</li>
     *   <li>chunk 2 → tok_a (weight 0.90) → 1st score</li>
     *   <li>chunk 3 → tok_e (weight 0.50) → 5th score</li>
     *   <li>chunk 4 → tok_c (weight 0.70) → 3rd score</li>
     * </ul>
     * Score order: passage 2 > passage 0 > passage 4 > passage 1 > passage 3.
     */
    private static final List<WeightedToken> SPARSE_QUERY_TOKENS = List.of(
        new WeightedToken("tok_a", 0.90f),
        new WeightedToken("tok_b", 0.80f),
        new WeightedToken("tok_c", 0.70f),
        new WeightedToken("tok_d", 0.60f),
        new WeightedToken("tok_e", 0.50f)
    );

    /** Token assigned to each chunk (index = chunk index, value = query token name). */
    private static final String[] CHUNK_SPARSE_TOKENS = { "tok_b", "tok_d", "tok_a", "tok_e", "tok_c" };

    public SemanticTextHighlighterTests(boolean useLegacyFormat) throws IOException {
        super(
            indexSettings(useLegacyFormat),
            Streams.readFully(SemanticTextHighlighterTests.class.getResourceAsStream("mappings-semantic_text.json")).utf8ToString(),
            buildSourceToParse(useLegacyFormat),
            buildDenseVectorQueryData(generatePassages())
        );
    }

    @ParametersFactory
    public static Iterable<Object[]> parameters() throws Exception {
        return List.of(new Object[] { true }, new Object[] { false });
    }

    public void testSparseVector() throws Exception {
        var fieldType = (SemanticTextFieldMapper.SemanticTextFieldType) mapperService.mappingLookup().getFieldType(SEMANTIC_FIELD_ELSER);
        SparseVectorQueryBuilder sparseQuery = new SparseVectorQueryBuilder(
            fieldType.getEmbeddingsField().fullPath(),
            SPARSE_QUERY_TOKENS,
            null,
            null,
            false,
            null
        );
        NestedQueryBuilder nestedQueryBuilder = new NestedQueryBuilder(fieldType.getChunksField().fullPath(), sparseQuery, ScoreMode.Max);
        var shardRequest = createShardSearchRequest(nestedQueryBuilder);

        String[] passages = generatePassages();
        // Score order: chunk2(tok_a,0.90) > chunk0(tok_b,0.80) > chunk4(tok_c,0.70) > chunk1(tok_d,0.60) > chunk3(tok_e,0.50)
        String[] expectedScorePassages = { passages[2], passages[0], passages[4], passages[1], passages[3] };
        for (int i = 0; i < expectedScorePassages.length; i++) {
            assertHighlightOneDoc(
                mapperService,
                createSearchExecutionContext(mapperService),
                shardRequest,
                sourceToParse,
                SEMANTIC_FIELD_ELSER,
                i + 1,
                HighlightBuilder.Order.SCORE,
                Arrays.copyOfRange(expectedScorePassages, 0, i + 1)
            );
        }

        String[] expectedOffsetPassages = passages.clone();
        assertHighlightOneDoc(
            mapperService,
            createSearchExecutionContext(mapperService),
            shardRequest,
            sourceToParse,
            SEMANTIC_FIELD_ELSER,
            expectedOffsetPassages.length,
            HighlightBuilder.Order.NONE,
            expectedOffsetPassages
        );
    }

    private static Settings indexSettings(boolean useLegacyFormat) {
        var indexVersion = useLegacyFormat ? getRandomCompatibleIndexVersion(true) : IndexVersion.current();
        return Settings.builder()
            .put(IndexMetadata.SETTING_VERSION_CREATED, indexVersion)
            .put(InferenceMetadataFieldsMapper.USE_LEGACY_SEMANTIC_TEXT_FORMAT.getKey(), useLegacyFormat)
            .build();
    }

    /**
     * Builds the test document source. The non-legacy format stores inference results in
     * {@code _inference_fields} with character-offset based chunk references; the legacy format
     * embeds the inference data directly on each semantic field with the chunk text inline.
     */
    private static SourceToParse buildSourceToParse(boolean useLegacyFormat) throws IOException {
        String[] passages = generatePassages();
        int[][] offsets = computeOffsets(passages);
        String fullText = buildFullText(passages);

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
        builder.field("body", fullText);

        if (useLegacyFormat) {
            writeDenseInferenceLegacy(builder, "field-semantic", ".multilingual-e5-small-elasticsearch", "text_embedding", passages);
            writeDenseInferenceLegacy(
                builder,
                "field-semantic-disk_bbq",
                ".multilingual-e5-small-elasticsearch",
                "text_embedding",
                passages
            );
            writeSparseInferenceLegacy(builder, "field-sparse-vector", ".elser-2-elasticsearch", passages);
        } else {
            builder.startObject("_inference_fields");
            writeDenseSemanticFieldInference(
                builder,
                "field-semantic",
                ".multilingual-e5-small-elasticsearch",
                "text_embedding",
                "body",
                offsets
            );
            writeDenseSemanticFieldInference(
                builder,
                "field-semantic-disk_bbq",
                ".multilingual-e5-small-elasticsearch",
                "text_embedding",
                "body",
                offsets
            );
            writeSparseInferenceNonLegacy(builder, "field-sparse-vector", ".elser-2-elasticsearch", "body", offsets, passages);
            builder.endObject(); // _inference_fields
        }

        builder.endObject();
        return new SourceToParse("0", BytesReference.bytes(builder), XContentType.JSON);
    }

    /**
     * Writes a dense semantic field's legacy inference section directly on the builder (not inside
     * {@code _inference_fields}). Legacy chunks carry the text inline instead of offsets.
     */
    private static void writeDenseInferenceLegacy(
        XContentBuilder builder,
        String fieldName,
        String inferenceId,
        String taskType,
        String[] passages
    ) throws IOException {
        builder.startObject(fieldName);
        builder.startObject("inference");
        builder.field("inference_id", inferenceId);
        builder.startObject("model_settings");
        builder.field("task_type", taskType);
        builder.field("dimensions", DIMS);
        builder.field("similarity", "cosine");
        builder.field("element_type", "float");
        builder.endObject(); // model_settings
        builder.startArray("chunks");
        for (int i = 0; i < CHUNK_K_VALUES.length; i++) {
            builder.startObject();
            builder.field("text", passages[i]);
            builder.array("embeddings", createChunkVector(CHUNK_K_VALUES[i]));
            builder.endObject();
        }
        builder.endArray(); // chunks
        builder.endObject(); // inference
        builder.endObject(); // fieldName
    }

    /**
     * Writes a sparse semantic field's legacy inference section directly on the builder.
     */
    private static void writeSparseInferenceLegacy(XContentBuilder builder, String fieldName, String inferenceId, String[] passages)
        throws IOException {
        builder.startObject(fieldName);
        builder.startObject("inference");
        builder.field("inference_id", inferenceId);
        builder.startObject("model_settings");
        builder.field("task_type", "sparse_embedding");
        builder.endObject(); // model_settings
        builder.startArray("chunks");
        for (int i = 0; i < CHUNK_K_VALUES.length; i++) {
            builder.startObject();
            builder.field("text", passages[i]);
            builder.startObject("embeddings");
            builder.field(CHUNK_SPARSE_TOKENS[i], 1.0f);
            builder.endObject();
            builder.endObject();
        }
        builder.endArray(); // chunks
        builder.endObject(); // inference
        builder.endObject(); // fieldName
    }

    /**
     * Writes a sparse semantic field's non-legacy inference section inside the open
     * {@code _inference_fields} object.
     */
    private static void writeSparseInferenceNonLegacy(
        XContentBuilder builder,
        String fieldName,
        String inferenceId,
        String chunkSourceField,
        int[][] offsets,
        String[] passages
    ) throws IOException {
        builder.startObject(fieldName);
        builder.startObject("inference");
        builder.field("inference_id", inferenceId);
        builder.startObject("model_settings");
        builder.field("task_type", "sparse_embedding");
        builder.endObject(); // model_settings
        builder.startObject("chunks");
        builder.startArray(chunkSourceField);
        for (int i = 0; i < CHUNK_K_VALUES.length; i++) {
            builder.startObject();
            builder.field("start_offset", offsets[i][0]);
            builder.field("end_offset", offsets[i][1]);
            builder.startObject("embeddings");
            builder.field(CHUNK_SPARSE_TOKENS[i], 1.0f);
            builder.endObject();
            builder.endObject();
        }
        builder.endArray();
        builder.endObject(); // chunks
        builder.endObject(); // inference
        builder.endObject(); // fieldName
    }
}
