/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.highlight;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.join.ScoreMode;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.elasticsearch.action.OriginalIndices;
import org.elasticsearch.action.search.SearchRequest;
import org.elasticsearch.cluster.metadata.IndexMetadata;
import org.elasticsearch.common.bytes.BytesArray;
import org.elasticsearch.common.bytes.BytesReference;
import org.elasticsearch.common.lucene.search.Queries;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.index.IndexVersion;
import org.elasticsearch.index.mapper.MapperService;
import org.elasticsearch.index.mapper.MapperServiceTestCase;
import org.elasticsearch.index.mapper.SourceToParse;
import org.elasticsearch.index.query.BoolQueryBuilder;
import org.elasticsearch.index.query.MatchAllQueryBuilder;
import org.elasticsearch.index.query.NestedQueryBuilder;
import org.elasticsearch.index.query.QueryBuilder;
import org.elasticsearch.index.query.SearchExecutionContext;
import org.elasticsearch.index.shard.ShardId;
import org.elasticsearch.plugins.Plugin;
import org.elasticsearch.search.SearchHit;
import org.elasticsearch.search.builder.SearchSourceBuilder;
import org.elasticsearch.search.fetch.FetchContext;
import org.elasticsearch.search.fetch.FetchSubPhase;
import org.elasticsearch.search.fetch.subphase.highlight.FieldHighlightContext;
import org.elasticsearch.search.fetch.subphase.highlight.HighlightBuilder;
import org.elasticsearch.search.fetch.subphase.highlight.SearchHighlightContext;
import org.elasticsearch.search.internal.AliasFilter;
import org.elasticsearch.search.internal.ShardSearchRequest;
import org.elasticsearch.search.lookup.Source;
import org.elasticsearch.search.rank.RankDoc;
import org.elasticsearch.search.vectors.KnnVectorQueryBuilder;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentFactory;
import org.elasticsearch.xcontent.XContentType;
import org.elasticsearch.xpack.inference.InferencePlugin;
import org.elasticsearch.xpack.inference.mapper.SemanticFieldMapper;
import org.mockito.Mockito;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.hamcrest.CoreMatchers.equalTo;
import static org.mockito.Mockito.mock;

public abstract class AbstractSemanticHighlighterTests extends MapperServiceTestCase {
    private static final String SEMANTIC_FIELD = "field-semantic";
    private static final String SEMANTIC_FIELD_DISK_BBQ = "field-semantic-disk_bbq";

    /**
     * Vector dimension matching the semantic field mappings.
     */
    static final int DIMS = 384;

    /**
     * Number of positive components (out of {@link #DIMS}) for each chunk vector.
     *
     * <p>The query vector is the all-positive unit vector {@code q = [1/sqrt(DIMS), ..., 1/sqrt(DIMS)]}.
     * Chunk {@code i} uses a unit vector whose first {@code CHUNK_K_VALUES[i]} components are
     * {@code +1/sqrt(DIMS)} and the remaining components are {@code -1/sqrt(DIMS)}.
     * The cosine similarity between the query and chunk {@code i} is {@code (2*k_i - DIMS) / DIMS},
     * and the Lucene score is {@code (1 + cosine) / 2}.
     *
     * <p>Score ordering (descending): chunk 1 (k=374) > chunk 3 (k=365) > chunk 0 (k=307) >
     * chunk 4 (k=269) > chunk 2 (k=230). Chunks 1 and 3 pass the 0.85 similarity threshold.
     *
     * <p>This design is intentionally non-sequential so that score order and document offset order differ,
     * exercising both {@link HighlightBuilder.Order#SCORE} and {@link HighlightBuilder.Order#NONE}.
     *
     * <p>Because all vector components have the same magnitude, 1-bit BBQ quantization preserves the
     * ordering exactly — the query's {@code +1/sqrt(DIMS)} components all quantize to the same sign,
     * and the score difference between chunks is captured purely by the number of positive bits.
     */
    static final int[] CHUNK_K_VALUES = { 307, 374, 230, 365, 269 };

    /**
     * Similarity score threshold used in the threshold-filtered highlight tests.
     * Corresponds to a minimum cosine of {@code 2 * SIMILARITY_THRESHOLD - 1 = 0.70}.
     */
    static final float SIMILARITY_THRESHOLD = 0.85f;

    final MapperService mapperService;
    final SourceToParse sourceToParse;
    final DenseVectorQueryData denseVectorQueryData;

    /**
     * Holds the pre-computed query vector and expected highlight results for dense-vector tests.
     */
    record DenseVectorQueryData(
        float[] queryVector,
        String[] expectedByScore,
        String[] expectedByOffset,
        String[] expectedWithSimilarityThreshold
    ) {}

    @SuppressWarnings("this-escape")
    public AbstractSemanticHighlighterTests(
        Settings settings,
        String mappings,
        SourceToParse sourceToParse,
        DenseVectorQueryData denseVectorQueryData
    ) throws IOException {
        this.mapperService = createMapperService(IndexMetadata.SETTING_INDEX_VERSION_CREATED.get(settings), settings, mappings);
        this.sourceToParse = sourceToParse;
        this.denseVectorQueryData = denseVectorQueryData;
    }

    @Override
    protected Collection<? extends Plugin> getPlugins() {
        return List.of(new InferencePlugin(Settings.EMPTY));
    }

    public void testDenseVector() throws Exception {
        float[] vector = denseVectorQueryData.queryVector();
        var fieldType = (SemanticFieldMapper.SemanticFieldType) mapperService.mappingLookup().getFieldType(SEMANTIC_FIELD);
        KnnVectorQueryBuilder knnQuery = new KnnVectorQueryBuilder(
            fieldType.getEmbeddingsField().fullPath(),
            vector,
            10,
            10,
            10f,
            null,
            null
        );
        NestedQueryBuilder nestedQueryBuilder = new NestedQueryBuilder(fieldType.getChunksField().fullPath(), knnQuery, ScoreMode.Max);
        var shardRequest = createShardSearchRequest(nestedQueryBuilder);

        String[] expectedScorePassages = denseVectorQueryData.expectedByScore();
        for (int i = 0; i < expectedScorePassages.length; i++) {
            assertHighlightOneDoc(
                mapperService,
                createSearchExecutionContext(mapperService),
                shardRequest,
                sourceToParse,
                SEMANTIC_FIELD,
                i + 1,
                HighlightBuilder.Order.SCORE,
                Arrays.copyOfRange(expectedScorePassages, 0, i + 1)
            );
        }

        String[] expectedOffsetPassages = denseVectorQueryData.expectedByOffset();
        assertHighlightOneDoc(
            mapperService,
            createSearchExecutionContext(mapperService),
            shardRequest,
            sourceToParse,
            SEMANTIC_FIELD,
            expectedOffsetPassages.length,
            HighlightBuilder.Order.NONE,
            expectedOffsetPassages
        );
    }

    public void testDenseVectorWithSimilarityThreshold() throws Exception {
        float[] vector = denseVectorQueryData.queryVector();
        var fieldType = (SemanticFieldMapper.SemanticFieldType) mapperService.mappingLookup().getFieldType(SEMANTIC_FIELD);

        KnnVectorQueryBuilder knnQuery = new KnnVectorQueryBuilder(
            fieldType.getEmbeddingsField().fullPath(),
            vector,
            10,
            10,
            10f,
            null,
            SIMILARITY_THRESHOLD
        );
        NestedQueryBuilder nestedQueryBuilder = new NestedQueryBuilder(fieldType.getChunksField().fullPath(), knnQuery, ScoreMode.Max);
        var shardRequest = createShardSearchRequest(nestedQueryBuilder);

        String[] expectedPassages = denseVectorQueryData.expectedWithSimilarityThreshold();
        assertHighlightOneDoc(
            mapperService,
            createSearchExecutionContext(mapperService),
            shardRequest,
            sourceToParse,
            SEMANTIC_FIELD,
            expectedPassages.length,
            HighlightBuilder.Order.SCORE,
            expectedPassages
        );
    }

    public void testDenseVectorWithDiskBBQandSimilarityThreshold() throws Exception {
        float[] vector = denseVectorQueryData.queryVector();
        var fieldType = (SemanticFieldMapper.SemanticFieldType) mapperService.mappingLookup().getFieldType(SEMANTIC_FIELD_DISK_BBQ);

        KnnVectorQueryBuilder knnQuery = new KnnVectorQueryBuilder(
            fieldType.getEmbeddingsField().fullPath(),
            vector,
            10,
            10,
            10f,
            null,
            SIMILARITY_THRESHOLD
        );
        NestedQueryBuilder nestedQueryBuilder = new NestedQueryBuilder(fieldType.getChunksField().fullPath(), knnQuery, ScoreMode.Max);
        var shardRequest = createShardSearchRequest(nestedQueryBuilder);

        String[] expectedPassages = denseVectorQueryData.expectedWithSimilarityThreshold();
        assertHighlightOneDoc(
            mapperService,
            createSearchExecutionContext(mapperService),
            shardRequest,
            sourceToParse,
            SEMANTIC_FIELD_DISK_BBQ,
            expectedPassages.length,
            HighlightBuilder.Order.SCORE,
            expectedPassages
        );
    }

    public void testDenseVectorWithDiskBBQ() throws Exception {
        float[] vector = denseVectorQueryData.queryVector();
        var fieldType = (SemanticFieldMapper.SemanticFieldType) mapperService.mappingLookup().getFieldType(SEMANTIC_FIELD_DISK_BBQ);

        KnnVectorQueryBuilder knnQuery = new KnnVectorQueryBuilder(
            fieldType.getEmbeddingsField().fullPath(),
            vector,
            10,
            10,
            10f,
            null,
            null
        );
        NestedQueryBuilder nestedQueryBuilder = new NestedQueryBuilder(fieldType.getChunksField().fullPath(), knnQuery, ScoreMode.Max);
        var shardRequest = createShardSearchRequest(nestedQueryBuilder);

        String[] expectedScorePassages = denseVectorQueryData.expectedByScore();
        for (int i = 0; i < expectedScorePassages.length; i++) {
            assertHighlightOneDoc(
                mapperService,
                createSearchExecutionContext(mapperService),
                shardRequest,
                sourceToParse,
                SEMANTIC_FIELD_DISK_BBQ,
                i + 1,
                HighlightBuilder.Order.SCORE,
                Arrays.copyOfRange(expectedScorePassages, 0, i + 1)
            );
        }

        String[] expectedOffsetPassages = denseVectorQueryData.expectedByOffset();
        assertHighlightOneDoc(
            mapperService,
            createSearchExecutionContext(mapperService),
            shardRequest,
            sourceToParse,
            SEMANTIC_FIELD_DISK_BBQ,
            expectedOffsetPassages.length,
            HighlightBuilder.Order.NONE,
            expectedOffsetPassages
        );
    }

    public void testNoSemanticField() throws Exception {
        float[] vector = denseVectorQueryData.queryVector();
        var fieldType = (SemanticFieldMapper.SemanticFieldType) mapperService.mappingLookup().getFieldType(SEMANTIC_FIELD);

        KnnVectorQueryBuilder knnQuery = new KnnVectorQueryBuilder(
            fieldType.getEmbeddingsField().fullPath(),
            vector,
            10,
            10,
            10f,
            null,
            null
        );
        var query = new BoolQueryBuilder().should(knnQuery).should(new MatchAllQueryBuilder());
        var shardRequest = createShardSearchRequest(query);
        var sourceToParse = new SourceToParse("0", new BytesArray("{}"), XContentType.JSON);
        assertHighlightOneDoc(
            mapperService,
            createSearchExecutionContext(mapperService),
            shardRequest,
            sourceToParse,
            SEMANTIC_FIELD,
            10,
            HighlightBuilder.Order.SCORE,
            new String[0]
        );
    }

    private MapperService createMapperService(IndexVersion indexVersion, Settings settings, String mappings) throws IOException {
        var mapperService = createMapperService(indexVersion, settings, mapping(b -> {}));
        merge(mapperService, mappings);
        return mapperService;
    }

    /**
     * Steps:
     * 1. Parse source into a document and create an index with it.
     * 2. Create a reader and searcher for the index.
     * 3. Create a FetchContext and HitContext for the document.
     * 4. Create a HighlightContext for the field and call the highlighter.
     * 5. Assert that the highlighted passages match the expected passages.
     * @param mapperService
     * @param execContext
     * @param request
     * @param source
     * @param fieldName
     * @param numFragments
     * @param order
     * @param expectedPassages
     * @throws Exception
     */
    static void assertHighlightOneDoc(
        MapperService mapperService,
        SearchExecutionContext execContext,
        ShardSearchRequest request,
        SourceToParse source,
        String fieldName,
        int numFragments,
        HighlightBuilder.Order order,
        String[] expectedPassages
    ) throws Exception {
        SemanticFieldMapper fieldMapper = (SemanticFieldMapper) mapperService.mappingLookup().getMapper(fieldName);
        var doc = mapperService.documentMapper().parse(source);
        assertNull(doc.dynamicMappingsUpdate());
        try (Directory dir = newDirectory()) {
            IndexWriterConfig iwc = newIndexWriterConfig(new StandardAnalyzer());
            RandomIndexWriter iw = new RandomIndexWriter(random(), dir, iwc);
            iw.addDocuments(doc.docs());
            try (DirectoryReader reader = wrapInMockESDirectoryReader(iw.getReader())) {
                IndexSearcher searcher = newSearcher(reader);
                iw.close();
                TopDocs topDocs = searcher.search(Queries.newNonNestedFilter(IndexVersion.current()), 1, Sort.INDEXORDER);
                assertThat(topDocs.totalHits.value(), equalTo(1L));
                int docID = topDocs.scoreDocs[0].doc;
                SemanticTextHighlighter highlighter = new SemanticTextHighlighter();
                var luceneQuery = execContext.toQuery(request.source().query()).query();
                FetchContext fetchContext = mock(FetchContext.class);
                Mockito.when(fetchContext.highlight()).thenReturn(new SearchHighlightContext(Collections.emptyList()));
                Mockito.when(fetchContext.query()).thenReturn(luceneQuery);
                Mockito.when(fetchContext.getSearchExecutionContext()).thenReturn(execContext);

                FetchSubPhase.HitContext hitContext = new FetchSubPhase.HitContext(
                    new SearchHit(docID),
                    getOnlyLeafReader(reader).getContext(),
                    docID,
                    Map.of(),
                    Source.fromBytes(source.source().originalBytes()),
                    new RankDoc(docID, Float.NaN, 0)
                );
                try {
                    var highlightContext = new HighlightBuilder().field(fieldName, 0, numFragments)
                        .order(order)
                        .highlighterType(SemanticTextHighlighter.NAME)
                        .build(execContext);

                    for (var fieldContext : highlightContext.fields()) {
                        FieldHighlightContext context = new FieldHighlightContext(
                            fieldName,
                            fieldContext,
                            fieldMapper.fieldType(),
                            fetchContext,
                            hitContext,
                            luceneQuery,
                            new HashMap<>()
                        );
                        var result = highlighter.highlight(context);
                        if (result == null) {
                            assertThat(expectedPassages.length, equalTo(0));
                        } else {
                            assertThat(result.fragments().length, equalTo(expectedPassages.length));
                            for (int i = 0; i < result.fragments().length; i++) {
                                assertThat(result.fragments()[i].string(), equalTo(expectedPassages[i]));
                            }
                        }
                    }
                } finally {
                    hitContext.hit().decRef();
                }
            }
        }
    }

    private static SearchRequest createSearchRequest(QueryBuilder queryBuilder) {
        SearchRequest request = new SearchRequest();
        request.source(new SearchSourceBuilder());
        request.allowPartialSearchResults(false);
        request.source().query(queryBuilder);
        return request;
    }

    static ShardSearchRequest createShardSearchRequest(QueryBuilder queryBuilder) {
        SearchRequest request = createSearchRequest(queryBuilder);
        return new ShardSearchRequest(OriginalIndices.NONE, request, new ShardId("index", "index", 0), 0, 1, AliasFilter.EMPTY, 1, 0, null);
    }

    // ---- Vector and passage generation utilities ----

    /**
     * Returns the unit query vector: all components equal to {@code 1/sqrt(DIMS)}.
     */
    static float[] createQueryVector() {
        float c = 1.0f / (float) Math.sqrt(DIMS);
        float[] v = new float[DIMS];
        Arrays.fill(v, c);
        return v;
    }

    /**
     * Returns a unit chunk vector with {@code k} positive and {@code DIMS - k} negative components,
     * each of magnitude {@code 1/sqrt(DIMS)}.
     */
    static float[] createChunkVector(int k) {
        float c = 1.0f / (float) Math.sqrt(DIMS);
        float[] v = new float[DIMS];
        for (int i = 0; i < DIMS; i++) {
            v[i] = i < k ? c : -c;
        }
        return v;
    }

    /** Returns short passage strings, one per entry in {@link #CHUNK_K_VALUES}. */
    static String[] generatePassages() {
        String[] passages = new String[CHUNK_K_VALUES.length];
        for (int i = 0; i < passages.length; i++) {
            passages[i] = "passage " + i;
        }
        return passages;
    }

    /**
     * Returns {@code [start, end]} character offsets for each passage in the full text produced by
     * {@link #buildFullText(String[])}, where passages are joined by {@code '\n'}.
     */
    static int[][] computeOffsets(String[] passages) {
        int[][] offsets = new int[passages.length][2];
        int pos = 0;
        for (int i = 0; i < passages.length; i++) {
            offsets[i][0] = pos;
            pos += passages[i].length();
            offsets[i][1] = pos;
            if (i < passages.length - 1) {
                pos++; // '\n' separator
            }
        }
        return offsets;
    }

    /** Joins passages with {@code '\n'} to form the source field text. */
    static String buildFullText(String[] passages) {
        return String.join("\n", passages);
    }

    /**
     * Builds the {@link DenseVectorQueryData} for the given passages by computing the score
     * ordering from {@link #CHUNK_K_VALUES} and applying the {@link #SIMILARITY_THRESHOLD}.
     */
    static DenseVectorQueryData buildDenseVectorQueryData(String[] passages) {
        int n = CHUNK_K_VALUES.length;

        // Sort chunk indices by descending k value (= descending score).
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Integer.compare(CHUNK_K_VALUES[b], CHUNK_K_VALUES[a]));

        String[] expectedByScore = new String[n];
        for (int i = 0; i < n; i++) {
            expectedByScore[i] = passages[order[i]];
        }

        // score = (1 + cos) / 2 >= threshold → cos >= 2*threshold - 1
        float minCos = 2 * SIMILARITY_THRESHOLD - 1;
        List<String> thresholdPassages = new ArrayList<>();
        for (int idx : order) { // already in descending score order
            float cos = (float) (2 * CHUNK_K_VALUES[idx] - DIMS) / DIMS;
            if (cos >= minCos) {
                thresholdPassages.add(passages[idx]);
            }
        }

        return new DenseVectorQueryData(createQueryVector(), expectedByScore, passages.clone(), thresholdPassages.toArray(String[]::new));
    }

    /**
     * Writes a dense semantic field's inference section into an already-open
     * {@code _inference_fields} object in {@code builder}.
     *
     * @param chunkSourceField the field name key used inside {@code chunks} (typically the text
     *                         source field such as {@code "field"} or {@code "body"})
     */
    static void writeDenseSemanticFieldInference(
        XContentBuilder builder,
        String semanticFieldName,
        String inferenceId,
        String taskType,
        String chunkSourceField,
        int[][] offsets
    ) throws IOException {
        builder.startObject(semanticFieldName);
        builder.startObject("inference");
        builder.field("inference_id", inferenceId);
        builder.startObject("model_settings");
        builder.field("task_type", taskType);
        builder.field("dimensions", DIMS);
        builder.field("similarity", "cosine");
        builder.field("element_type", "float");
        builder.endObject(); // model_settings
        builder.startObject("chunks");
        builder.startArray(chunkSourceField);
        for (int i = 0; i < CHUNK_K_VALUES.length; i++) {
            builder.startObject();
            builder.field("start_offset", offsets[i][0]);
            builder.field("end_offset", offsets[i][1]);
            builder.array("embeddings", createChunkVector(CHUNK_K_VALUES[i]));
            builder.endObject();
        }
        builder.endArray();
        builder.endObject(); // chunks
        builder.endObject(); // inference
        builder.endObject(); // semanticFieldName
    }

    /**
     * Builds a {@link SourceToParse} containing the full text in {@code textFieldName} and
     * inference data for two dense semantic fields ({@code field-semantic} and
     * {@code field-semantic-disk_bbq}) inside {@code _inference_fields}.
     *
     * @param textFieldName  the text source field (e.g. {@code "field"} or {@code "body"})
     * @param inferenceId    the inference endpoint id stored in model metadata
     * @param taskType       the model task type (e.g. {@code "embedding"} or {@code "text_embedding"})
     */
    static SourceToParse buildDenseFieldSourceToParse(String textFieldName, String inferenceId, String taskType) throws IOException {
        String[] passages = generatePassages();
        int[][] offsets = computeOffsets(passages);

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
        builder.field(textFieldName, buildFullText(passages));
        builder.startObject("_inference_fields");
        writeDenseSemanticFieldInference(builder, "field-semantic", inferenceId, taskType, textFieldName, offsets);
        writeDenseSemanticFieldInference(builder, "field-semantic-disk_bbq", inferenceId, taskType, textFieldName, offsets);
        builder.endObject(); // _inference_fields
        builder.endObject();

        return new SourceToParse("0", BytesReference.bytes(builder), XContentType.JSON);
    }
}
