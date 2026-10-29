/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.esql.plugin;

import com.carrotsearch.randomizedtesting.annotations.Name;
import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;

import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.index.mapper.vectors.DenseVectorFieldMapper.ElementType;
import org.elasticsearch.search.vectors.DenseVectorQueryBuilder;
import org.elasticsearch.xpack.esql.EsqlTestUtils;
import org.elasticsearch.xpack.esql.action.AbstractEsqlIntegTestCase;
import org.junit.Before;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;

import static org.elasticsearch.test.hamcrest.ElasticsearchAssertions.assertAcked;
import static org.elasticsearch.test.hamcrest.ElasticsearchAssertions.assertResponse;

/** Exercises pushed-down and per-row KNN searches over vectors stored only in doc values. */
public class KnnNonIndexedFunctionIT extends AbstractEsqlIntegTestCase {
    private final ElementType elementType;

    @ParametersFactory
    public static Iterable<Object[]> parameters() {
        return Arrays.stream(ElementType.values()).map(type -> new Object[] { type }).toList();
    }

    public KnnNonIndexedFunctionIT(@Name("elementType") ElementType elementType) {
        this.elementType = elementType;
    }

    private float[] queryVector() {
        return elementType == ElementType.BIT ? new float[] { 1 } : new float[] { 1, 0, 0 };
    }

    @Before
    public void setupVectors() throws Exception {
        createVectorIndex("vectors", false);
        indexRandom(
            true,
            prepareIndex("vectors").setId("1").setSource("id", 1, "value", 1.0, "emb", queryVector()),
            prepareIndex("vectors").setId("2")
                .setSource("id", 2, "value", 2.0, "emb", elementType == ElementType.BIT ? new float[] { 2 } : new float[] { 0, 1, 0 }),
            prepareIndex("vectors").setId("3")
                .setSource("id", 3, "value", 3.0, "emb", elementType == ElementType.BIT ? new float[] { -2 } : new float[] { -1, 0, 0 }),
            prepareIndex("vectors").setId("4").setSource("id", 4, "value", 4.0)
        );
    }

    private void createVectorIndex(String index, boolean indexed) {
        String indexOptions = indexed
            ? ", \"similarity\": \""
                + (elementType == ElementType.BIT ? "l2_norm" : "cosine")
                + "\", \"index_options\": {\"type\": \"hnsw\"}"
            : "";
        assertAcked(
            prepareCreate(index).setSettings(
                Settings.builder().put("number_of_shards", randomIntBetween(1, 2)).put("number_of_replicas", 0)
            ).setMapping(String.format(Locale.ROOT, """
                {"properties": {
                  "id": {"type": "integer"},
                  "value": {"type": "double"},
                  "emb": {"type": "dense_vector", "element_type": "%s", "dims": %d, "index": %s %s},
                  "emb_alias": {"type": "alias", "path": "emb"}
                }}
                """, elementType.toString().toLowerCase(Locale.ROOT), elementType == ElementType.BIT ? 8 : 3, indexed, indexOptions))
        );
    }

    public void testScoresMatchDenseVectorQuery() {
        assertScoresMatchDenseVectorQuery(false, randomFrom(1.0f, 2.0f));
    }

    public void testNonPushedScoresMatchDenseVectorQuery() {
        assertScoresMatchDenseVectorQuery(true, 1.0f);
    }

    private void assertScoresMatchDenseVectorQuery(boolean evaluated, float boost) {
        assertResponse(
            client().prepareSearch("vectors").setQuery(new DenseVectorQueryBuilder("emb", queryVector(), null, false).boost(boost)),
            expected -> {
                try (
                    var actual = run(
                        "FROM vectors METADATA _score"
                            + (evaluated ? " | EVAL value = value * 1.1" : "")
                            + " | WHERE KNN(emb, "
                            + Arrays.toString(queryVector())
                            + ", {\"boost\": "
                            + boost
                            + "})"
                            + (evaluated ? " OR value > 100.0" : "")
                            + " | SORT _score DESC, id | KEEP id, _score | LIMIT 10"
                    )
                ) {
                    var rows = EsqlTestUtils.getValuesList(actual);
                    assertEquals(3, rows.size());
                    var hits = expected.getHits().getHits();
                    for (int i = 0; i < hits.length; i++) {
                        assertEquals(Integer.parseInt(hits[i].getId()), rows.get(i).get(0));
                        assertEquals(hits[i].getScore(), (Double) rows.get(i).get(1), 0.000001);
                    }
                }
            }
        );
    }

    public void testThreshold() {
        assertRows("FROM vectors | WHERE KNN(emb, " + Arrays.toString(queryVector()) + ", {\"similarity\": 0.9})", List.of(List.of(1)));
    }

    public void testComputedDisjunction() {
        assertRows(
            "FROM vectors | EVAL value = value * 1.1 | WHERE KNN(emb, "
                + Arrays.toString(queryVector())
                + ", {\"similarity\": 0.9}) OR value > 3.0",
            List.of(List.of(1), List.of(3), List.of(4))
        );
    }

    public void testComputedPrefilter() {
        assertRows(
            "FROM vectors | EVAL value = value * 1.1 | WHERE KNN(emb, " + Arrays.toString(queryVector()) + ") AND value > 2.0",
            List.of(List.of(2), List.of(3))
        );
    }

    public void testPrefilterAndFieldAlias() {
        assertRows(
            "FROM vectors | RENAME emb_alias AS vector | WHERE KNN(vector, " + Arrays.toString(queryVector()) + ") AND id > 1",
            List.of(List.of(2), List.of(3))
        );
    }

    public void testMixedIndices() throws Exception {
        createVectorIndex("vectors_indexed", true);
        assertAcked(prepareCreate("vectors_unmapped").setMapping("id", "type=integer"));
        indexRandom(
            true,
            prepareIndex("vectors_indexed").setSource("id", 5, "value", 5.0, "emb", queryVector()),
            prepareIndex("vectors_unmapped").setSource("id", 6)
        );
        assertRows(
            "FROM vectors* | WHERE KNN(emb, " + Arrays.toString(queryVector()) + ", {\"similarity\": 0.9})",
            List.of(List.of(1), List.of(5))
        );
    }

    private void assertRows(String query, List<List<Object>> expected) {
        try (var response = run(query + " | SORT id | KEEP id | LIMIT 10")) {
            assertEquals(expected, EsqlTestUtils.getValuesList(response));
        }
    }
}
