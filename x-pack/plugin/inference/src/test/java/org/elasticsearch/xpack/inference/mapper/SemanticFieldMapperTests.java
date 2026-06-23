/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.inference.mapper;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;

import org.elasticsearch.cluster.metadata.IndexMetadata;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.core.CheckedConsumer;
import org.elasticsearch.index.IndexVersion;
import org.elasticsearch.index.IndexVersions;
import org.elasticsearch.index.mapper.MappedFieldType;
import org.elasticsearch.index.mapper.Mapper;
import org.elasticsearch.index.mapper.MapperParsingException;
import org.elasticsearch.index.mapper.MapperService;
import org.elasticsearch.index.mapper.vectors.DenseVectorFieldMapper;
import org.elasticsearch.inference.InferenceService;
import org.elasticsearch.inference.MinimalServiceSettings;
import org.elasticsearch.inference.SimilarityMeasure;
import org.elasticsearch.license.License;
import org.elasticsearch.test.index.IndexVersionUtils;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xpack.inference.services.elastic.ElasticInferenceService;
import org.junit.BeforeClass;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.elasticsearch.inference.TaskType.EMBEDDING;
import static org.elasticsearch.xpack.inference.mapper.SemanticFieldMapper.CONTENT_TYPE;
import static org.elasticsearch.xpack.inference.mapper.SemanticTextField.INFERENCE_ID_FIELD;
import static org.elasticsearch.xpack.inference.mapper.SemanticTextField.SEARCH_INFERENCE_ID_FIELD;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.instanceOf;
import static org.mockito.Mockito.mock;

public class SemanticFieldMapperTests extends AbstractSemanticMapperTestCase {
    private static final String INFERENCE_ID = "inference-id";
    private static final String SEARCH_INFERENCE_ID = "search-inference-id";
    private static final String FIELD_NAME = "my_field";

    public SemanticFieldMapperTests(License.OperationMode operationMode) {
        super(operationMode);
    }

    @BeforeClass
    public static void checkFeatureFlag() {
        assumeTrue("Semantic field feature flag is not enabled", SemanticFieldMapper.SEMANTIC_FIELD_FEATURE_FLAG.isEnabled());
    }

    @Override
    protected void registerDefaultEndpoints() {
        registerMultiModalEisEndpoint();
    }

    @ParametersFactory
    public static Iterable<Object[]> parameters() throws Exception {
        return List.of(new Object[] { License.OperationMode.BASIC }, new Object[] { License.OperationMode.ENTERPRISE });
    }

    private void registerMultiModalEisEndpoint() {
        globalModelRegistry.putDefaultIdIfAbsent(
            new InferenceService.DefaultConfigId(
                INFERENCE_ID,
                new MinimalServiceSettings(
                    ElasticInferenceService.NAME,
                    EMBEDDING,
                    1024,
                    SimilarityMeasure.COSINE,
                    DenseVectorFieldMapper.ElementType.FLOAT
                ),
                mock(InferenceService.class)
            )
        );
    }

    public void testSemanticFieldNotSupportedOnOldIndices() throws IOException {
        IndexVersion oldVersion = IndexVersionUtils.randomPreviousCompatibleVersion(IndexVersions.SEMANTIC_FIELD_TYPE);
        Settings settings = Settings.builder().put(IndexMetadata.SETTING_INDEX_VERSION_CREATED.getKey(), oldVersion).build();

        var ex = expectThrows(MapperParsingException.class, () -> createMapperService(oldVersion, settings, mapping(b -> {
            b.startObject(FIELD_NAME);
            b.field("type", CONTENT_TYPE);
            b.field(INFERENCE_ID_FIELD, "test_model");
            b.endObject();
        })));
        assertSemanticFieldVersionNotSupported(ex);
    }

    public void testSemanticFieldSupportedOnNewIndices() throws IOException {
        IndexVersion newVersion = IndexVersionUtils.randomVersionOnOrAfter(IndexVersions.SEMANTIC_FIELD_TYPE);
        Settings settings = Settings.builder().put(IndexMetadata.SETTING_INDEX_VERSION_CREATED.getKey(), newVersion).build();

        // model_settings provided so the model registry is not consulted for unknown endpoints
        var mapperService = createMapperService(newVersion, settings, mapping(b -> writeSemanticField(b, FIELD_NAME, "test_model", null)));
        assertSemanticFieldMapper(mapperService, FIELD_NAME, "test_model", "test_model");
    }

    public void testSemanticFieldMappingUpdateNotSupportedOnOldIndices() throws IOException {
        IndexVersion oldVersion = IndexVersionUtils.randomPreviousCompatibleVersion(IndexVersions.SEMANTIC_FIELD_TYPE);
        Settings settings = Settings.builder().put(IndexMetadata.SETTING_INDEX_VERSION_CREATED.getKey(), oldVersion).build();

        var mapperService = createMapperService(oldVersion, settings, mapping(b -> {}));

        var ex = expectThrows(MapperParsingException.class, () -> merge(mapperService, mapping(b -> {
            b.startObject(FIELD_NAME);
            b.field("type", CONTENT_TYPE);
            b.field(INFERENCE_ID_FIELD, "test_model");
            b.endObject();
        })));
        assertSemanticFieldVersionNotSupported(ex);
    }

    public void testSemanticFieldMappingUpdateSupportedOnNewIndices() throws IOException {
        IndexVersion newVersion = IndexVersionUtils.randomVersionOnOrAfter(IndexVersions.SEMANTIC_FIELD_TYPE);
        Settings settings = Settings.builder().put(IndexMetadata.SETTING_INDEX_VERSION_CREATED.getKey(), newVersion).build();

        var mapperService = createMapperService(newVersion, settings, mapping(b -> {}));
        // model_settings provided so the model registry is not consulted for unknown endpoints
        merge(mapperService, mapping(b -> writeSemanticField(b, FIELD_NAME, "test_model", null)));

        assertSemanticFieldMapper(mapperService, FIELD_NAME, "test_model", "test_model");
    }

    public void testSetInferenceEndpoints() throws IOException {
        {
            MapperService mapperService = createMapperService(
                fieldMapping(b -> b.field("type", CONTENT_TYPE).field(INFERENCE_ID_FIELD, INFERENCE_ID))
            );
            assertSemanticFieldMapper(mapperService, "field", INFERENCE_ID, INFERENCE_ID);
        }
        {
            MapperService mapperService = createMapperService(
                fieldMapping(
                    b -> b.field("type", CONTENT_TYPE)
                        .field(INFERENCE_ID_FIELD, INFERENCE_ID)
                        .field(SEARCH_INFERENCE_ID_FIELD, SEARCH_INFERENCE_ID)
                )
            );
            assertSemanticFieldMapper(mapperService, "field", INFERENCE_ID, SEARCH_INFERENCE_ID);
        }
    }

    private static void assertSemanticFieldVersionNotSupported(MapperParsingException ex) {
        assertThat(ex.getMessage(), containsString("[" + CONTENT_TYPE + "]"));
        assertThat(ex.getMessage(), containsString("is not supported on indices created before version"));
        assertThat(ex.getMessage(), containsString(IndexVersions.SEMANTIC_FIELD_TYPE.toString()));
    }

    private static void assertSemanticFieldMapper(
        MapperService mapperService,
        String fieldName,
        String expectedInferenceId,
        String expectedSearchInferenceId
    ) {
        Mapper mapper = mapperService.mappingLookup().getMapper(fieldName);
        assertThat(mapper, instanceOf(SemanticFieldMapper.class));
        SemanticFieldMapper semanticFieldMapper = (SemanticFieldMapper) mapper;
        assertThat(semanticFieldMapper.fieldType().getInferenceId(), equalTo(expectedInferenceId));
        assertThat(semanticFieldMapper.fieldType().getSearchInferenceId(), equalTo(expectedSearchInferenceId));
    }

    /**
     * Writes a semantic field mapping with explicit model_settings so mapper construction does not
     * need to resolve the inference endpoint from the model registry.
     */
    private static void writeSemanticField(XContentBuilder b, String fieldName, String inferenceId, String searchInferenceId)
        throws IOException {
        b.startObject(fieldName);
        b.field("type", CONTENT_TYPE);
        b.field(INFERENCE_ID_FIELD, inferenceId);
        if (searchInferenceId != null) {
            b.field(SEARCH_INFERENCE_ID_FIELD, searchInferenceId);
        }
        writeDefaultModelSettings(b);
        b.endObject();
    }

    private static void writeDefaultModelSettings(XContentBuilder b) throws IOException {
        b.startObject("model_settings");
        b.field("task_type", "embedding");
        b.field("dimensions", 128);
        b.field("similarity", "cosine");
        b.field("element_type", "float");
        b.endObject();
    }

    @Override
    protected void minimalMapping(XContentBuilder b) throws IOException {
        b.field("type", CONTENT_TYPE);
        b.field(INFERENCE_ID_FIELD, INFERENCE_ID);
    }

    @Override
    public MappedFieldType getMappedFieldType() {
        return new SemanticFieldMapper.SemanticFieldType("field", INFERENCE_ID, null, null, null, null, null, Map.of());
    }

    @Override
    protected Object getSampleObjectForDocument() {
        // Multimodal semantic fields accept typed inputs (text/image) as objects.
        return Map.of("type", "image", "value", "data:image/jpeg;base64,Y2F0IG9uIGEgd2luZG93c2lsbA==");
    }

    @Override
    protected void assertSearchable(MappedFieldType fieldType) {
        assertThat(fieldType, instanceOf(SemanticFieldMapper.SemanticFieldType.class));
        assertTrue(fieldType.isSearchable());
    }

    @Override
    protected Set<IndexVersion> getSupportedVersions() {
        return IndexVersionUtils.allReleasedVersions()
            .stream()
            .filter(v -> v.onOrAfter(IndexVersions.SEMANTIC_FIELD_TYPE))
            .collect(Collectors.toSet());
    }

    @Override
    protected IndexVersion boostNotAllowedIndexVersion() {
        return IndexVersions.SEMANTIC_FIELD_TYPE;
    }

    public void testCustomInferenceIdIsMandatory() {
        Exception e = expectThrows(
            MapperParsingException.class,
            () -> createMapperService(fieldMapping(b -> b.field("type", CONTENT_TYPE)))
        );

        assertThat(e.getMessage(), containsString("[inference_id] on mapper [field] of type [" + CONTENT_TYPE + "] must not be empty"));
    }

    public void testInvalidInferenceEndpoints() {
        assertInvalidInferenceEndpoint(
            b -> b.field("type", CONTENT_TYPE).field(INFERENCE_ID_FIELD, (String) null),
            "[inference_id] on mapper [field] of type [" + CONTENT_TYPE + "] must not have a [null] value"
        );
        assertInvalidInferenceEndpoint(
            b -> b.field("type", CONTENT_TYPE).field(INFERENCE_ID_FIELD, ""),
            "[inference_id] on mapper [field] of type [" + CONTENT_TYPE + "] must not be empty"
        );
        assertInvalidInferenceEndpoint(
            b -> b.field("type", CONTENT_TYPE).field(INFERENCE_ID_FIELD, INFERENCE_ID).field(SEARCH_INFERENCE_ID_FIELD, (String) null),
            "[search_inference_id] on mapper [field] of type [" + CONTENT_TYPE + "] must not have a [null] value"
        );
        assertInvalidInferenceEndpoint(
            b -> b.field("type", CONTENT_TYPE).field(INFERENCE_ID_FIELD, INFERENCE_ID).field(SEARCH_INFERENCE_ID_FIELD, ""),
            "[search_inference_id] on mapper [field] of type [" + CONTENT_TYPE + "] must not be empty"
        );
    }

    private void assertInvalidInferenceEndpoint(CheckedConsumer<XContentBuilder, IOException> mapping, String expectedMessage) {
        Exception e = expectThrows(MapperParsingException.class, () -> createMapperService(fieldMapping(mapping)));
        assertThat(e.getMessage(), containsString(expectedMessage));
    }
}
