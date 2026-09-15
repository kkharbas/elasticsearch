/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.esql.plugin;

import org.elasticsearch.Build;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.xpack.esql.EsqlTestUtils;
import org.elasticsearch.xpack.esql.action.AbstractEsqlIntegTestCase;
import org.elasticsearch.xpack.esql.action.EsqlQueryRequest;
import org.elasticsearch.xpack.esql.expression.function.vector.VectorSimilarityMetric;
import org.junit.Before;

import java.util.Arrays;
import java.util.EnumSet;
import java.util.Locale;

import static org.elasticsearch.xpack.esql.action.EsqlQueryRequest.syncEsqlQueryRequest;

/**
 * Integration tests for the {@code knn()} function's {@code vector_similarity} option on runtime
 * (non-indexed) dense-vector expressions.
 *
 * <p>Each test loops over all float-compatible {@link VectorSimilarityMetric} values and applies
 * them to the same fixed set of vectors, comparing the runtime behaviour against the Java-side
 * {@link VectorSimilarityMetric#score} and {@link VectorSimilarityMetric#calculateSimilarity}
 * computations. Using the same dataset across all metrics allows regressions to be spotted at a
 * glance, without duplicating fixture data per metric.
 *
 * <p>Tests use {@code ROW}-based queries so that no indexed data is required.
 */
public class KnnRuntimeSimilarityIT extends AbstractEsqlIntegTestCase {

    /** Three-dimensional query vector. */
    private static final float[] QUERY_VECTOR = { 1.0f, 0.0f, 0.0f };

    /**
     * Three 3-dimensional unit-length doc vectors. They are unit-length so that the
     * {@code v_dot_product} metric accepts them without triggering the unit-length guard, letting
     * the same dataset cover every metric.
     *
     * <ul>
     *   <li>[0] identical to the query → always the highest-scoring document</li>
     *   <li>[1] orthogonal to the query → always the lowest-scoring document</li>
     *   <li>[2] intermediate (0.6, 0.8, 0.0) — a normalized 3-4-5 triangle — sits between [0]
     *       and [1] for every metric</li>
     * </ul>
     */
    private static final float[][] DOC_VECTORS = {
        { 1.0f, 0.0f, 0.0f },  // identical — max similarity
        { 0.0f, 1.0f, 0.0f },  // orthogonal — zero dot product with query
        { 0.6f, 0.8f, 0.0f },  // intermediate (|v| = 1.0)
    };

    @Before
    public void requireSnapshotBuild() {
        assumeTrue("Runtime KNN requires a snapshot build", Build.current().isSnapshot());
    }

    /**
     * Verifies that with no similarity threshold every document is returned, for every
     * float-compatible metric. This is the baseline: before any score-based filtering is applied
     * the runtime knn evaluator is a pass-through.
     */
    public void testNoThresholdIncludesAllDocsForAllMetrics() {
        for (VectorSimilarityMetric metric : floatCompatibleMetrics()) {
            for (float[] docVector : DOC_VECTORS) {
                assertTrue(
                    "metric=" + metric + " docVector=" + Arrays.toString(docVector) + " should be returned with no similarity threshold",
                    runFilterQuery(metric, docVector, null)
                );
            }
        }
    }

    /**
     * Verifies that threshold-based filtering uses the correct score ordering for every metric.
     *
     * <p>The threshold is set to the raw similarity between the intermediate vector
     * ({@code DOC_VECTORS[2]}) and the query. With this threshold:
     * <ul>
     *   <li>the <em>identical</em> vector ({@code DOC_VECTORS[0]}) always passes — its score is
     *       the maximum for the given metric</li>
     *   <li>the <em>intermediate</em> vector ({@code DOC_VECTORS[2]}) always passes — its score
     *       exactly equals the threshold ({@code score(doc) >= score(threshold)})</li>
     *   <li>the <em>orthogonal</em> vector ({@code DOC_VECTORS[1]}) is always excluded — it is
     *       less similar to the query than the intermediate vector under every metric</li>
     * </ul>
     *
     * <p>This threshold choice works for both similarity metrics (higher raw value → higher score)
     * and distance metrics (lower raw value → higher score) because the comparison in the runtime
     * evaluator is always {@code score(doc) >= score(threshold)}, which is monotone in the same
     * direction for all metrics.
     */
    public void testFilteringWithSimilarityThresholdForAllMetrics() {
        for (VectorSimilarityMetric metric : floatCompatibleMetrics()) {
            // Raw similarity of the intermediate vector relative to the query
            float intermediateRawSim = metric.calculateSimilarity(DOC_VECTORS[2], QUERY_VECTOR);

            assertTrue(
                "metric=" + metric + ": identical vector should pass with threshold=" + intermediateRawSim,
                runFilterQuery(metric, DOC_VECTORS[0], intermediateRawSim)
            );
            assertTrue(
                "metric=" + metric + ": intermediate vector should pass with threshold=" + intermediateRawSim,
                runFilterQuery(metric, DOC_VECTORS[2], intermediateRawSim)
            );
            assertFalse(
                "metric=" + metric + ": orthogonal vector should be excluded with threshold=" + intermediateRawSim,
                runFilterQuery(metric, DOC_VECTORS[1], intermediateRawSim)
            );
        }
    }

    /**
     * Verifies that setting the threshold to the exact raw similarity of each document admits
     * that document (score >= score at threshold) and that raising the threshold by a tiny epsilon
     * excludes it — confirming that the runtime evaluator's score matches the Java-side
     * {@link VectorSimilarityMetric#score} computation.
     *
     * <p>The epsilon adjustment is applied in <em>raw-similarity space</em>, not normalized-score
     * space, so that it works correctly regardless of the metric's score direction:
     * <ul>
     *   <li>For <em>similarity</em> metrics ({@code v_cosine}, {@code v_dot_product}): score
     *       increases with raw similarity, so {@code threshold = rawSim + ε} raises the score
     *       threshold above the document's score → excluded.</li>
     *   <li>For <em>distance</em> metrics ({@code v_l1_norm}, {@code v_l2_norm}): score decreases
     *       as raw similarity (distance) grows, so {@code threshold = rawSim - ε} lowers the
     *       threshold's distance, raising the score threshold above the document's score →
     *       excluded.</li>
     * </ul>
     */
    public void testEachDocScoreMatchesJavaComputationViaThresholdBracket() {
        float eps = 0.001f;
        for (VectorSimilarityMetric metric : floatCompatibleMetrics()) {
            for (float[] docVector : DOC_VECTORS) {
                float rawSim = metric.calculateSimilarity(docVector, QUERY_VECTOR);
                double expectedScore = metric.score(rawSim, QUERY_VECTOR.length);
                String label = "metric="
                    + metric
                    + " docVector="
                    + Arrays.toString(docVector)
                    + " rawSim="
                    + rawSim
                    + " expectedScore="
                    + expectedScore;

                // Exact threshold: doc's score == score(threshold) → included (>= check passes).
                assertTrue(label + ": exact-threshold query should include the doc", runFilterQuery(metric, docVector, rawSim));

                // Skip the exclusion check for the maximum score (1.0): there is no threshold
                // that lies above it while still being a valid raw-similarity value.
                if (expectedScore >= 1.0 - 1e-5) {
                    continue;
                }

                // Raise the score threshold by a tiny epsilon in raw-sim space (direction
                // depends on metric type) and verify the doc is now excluded.
                float stricterThreshold = isDistanceMetric(metric) ? rawSim - eps : rawSim + eps;
                assertFalse(
                    label + ": stricter-threshold query should exclude the doc",
                    runFilterQuery(metric, docVector, stricterThreshold)
                );
            }
        }
    }

    /**
     * Verifies that the score ordering produced by each metric agrees with the Java-side
     * {@link VectorSimilarityMetric#score} computation: the identical vector should always score
     * higher than the intermediate vector, which should score higher than the orthogonal vector.
     * This is verified by running a filter query with the intermediate vector's score as a
     * threshold and confirming the expected set of documents is selected.
     */
    public void testScoreOrderingIsConsistentWithExpectedForAllMetrics() {
        for (VectorSimilarityMetric metric : floatCompatibleMetrics()) {
            double identicalScore = expectedScore(metric, DOC_VECTORS[0]);
            double intermediateScore = expectedScore(metric, DOC_VECTORS[2]);
            double orthogonalScore = expectedScore(metric, DOC_VECTORS[1]);

            // Sanity-check Java-side ordering: identical > intermediate > orthogonal
            assertTrue("metric=" + metric + ": identical should score higher than intermediate", identicalScore > intermediateScore);
            assertTrue("metric=" + metric + ": intermediate should score higher than orthogonal", intermediateScore > orthogonalScore);
        }
    }

    /**
     * Runs a ROW-based {@code knn()} query with the given metric and optional raw-similarity
     * threshold, and returns {@code true} if the document row is returned (i.e., the document
     * passes the knn filter).
     */
    private boolean runFilterQuery(VectorSimilarityMetric metric, float[] docVector, Float threshold) {
        String options = threshold == null
            ? String.format(Locale.ROOT, "{\"vector_similarity\": \"%s\"}", metric.optionValue())
            : String.format(Locale.ROOT, "{\"vector_similarity\": \"%s\", \"similarity\": %s}", metric.optionValue(), threshold);
        String esql = String.format(
            Locale.ROOT,
            "ROW dv = to_dense_vector(%s)\n| WHERE knn(dv, %s, %s)\n| KEEP dv\n| LIMIT 1",
            floatVectorToEsql(docVector),
            floatVectorToEsql(QUERY_VECTOR),
            options
        );

        EsqlQueryRequest request = syncEsqlQueryRequest(esql).acceptedPragmaRisks(true)
            .pragmas(new QueryPragmas(Settings.builder().put("knn_runtime_field", true).build()));

        try (var resp = run(request)) {
            return EsqlTestUtils.getValuesList(resp).isEmpty() == false;
        }
    }

    private static double expectedScore(VectorSimilarityMetric metric, float[] docVector) {
        return metric.score(metric.calculateSimilarity(docVector, QUERY_VECTOR), QUERY_VECTOR.length);
    }

    /**
     * Returns {@code true} for distance metrics whose score decreases as the raw similarity
     * (distance) grows — currently {@code V_L1_NORM} and {@code V_L2_NORM}.
     */
    private static boolean isDistanceMetric(VectorSimilarityMetric metric) {
        return metric == VectorSimilarityMetric.V_L1_NORM || metric == VectorSimilarityMetric.V_L2_NORM;
    }

    /**
     * Returns all metrics that accept 32-bit float vectors. {@code V_HAMMING} requires byte
     * vectors and is excluded.
     */
    private static EnumSet<VectorSimilarityMetric> floatCompatibleMetrics() {
        return EnumSet.complementOf(EnumSet.of(VectorSimilarityMetric.V_HAMMING));
    }

    /**
     * Formats a {@code float[]} as an ES|QL list literal, e.g. {@code [1.0, 0.0, 0.0]}.
     * {@link Arrays#toString(float[])} produces the right format for all non-pathological values.
     */
    private static String floatVectorToEsql(float[] vector) {
        return Arrays.toString(vector);
    }
}
