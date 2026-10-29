/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.esql.plugin;

import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.Query;
import org.elasticsearch.index.mapper.MapperService;
import org.elasticsearch.index.mapper.MapperServiceTestCase;
import org.elasticsearch.index.query.QueryBuilder;
import org.elasticsearch.search.vectors.DenseVectorQueryBuilder;
import org.elasticsearch.search.vectors.ExactKnnQueryBuilder;
import org.elasticsearch.search.vectors.KnnVectorQueryBuilder;
import org.elasticsearch.search.vectors.VectorData;
import org.elasticsearch.search.vectors.VectorSimilarityQuery;

import java.io.IOException;

import static org.elasticsearch.index.query.QueryBuilders.termQuery;
import static org.hamcrest.Matchers.containsString;

/** Checks that exact vector search is enabled by the ES|QL context without changing Query DSL kNN behavior. */
public class NonIndexedVectorQueryTests extends MapperServiceTestCase {
    private static final float[] VECTOR = { 1, 0, 0 };

    public void testKnn() throws IOException {
        assertNonIndexedQuery(new KnnVectorQueryBuilder("vector", VECTOR, 1, 10, null, null, null), null);
    }

    public void testKnnWithThreshold() throws IOException {
        assertNonIndexedQuery(new KnnVectorQueryBuilder("vector", VECTOR, 1, 10, null, null, 0.9f), 0.9f);
    }

    public void testExactKnn() throws IOException {
        assertNonIndexedQuery(new ExactKnnQueryBuilder(VectorData.fromFloats(VECTOR), "vector", null), null);
    }

    public void testExactKnnWithThreshold() throws IOException {
        assertNonIndexedQuery(new ExactKnnQueryBuilder(VectorData.fromFloats(VECTOR), "vector", 0.9f), 0.9f);
    }

    private void assertNonIndexedQuery(QueryBuilder query, Float threshold) throws IOException {
        MapperService mapper = mapper();
        var context = createSearchExecutionContext(mapper);
        var error = expectThrows(IllegalArgumentException.class, () -> query.toQuery(context));
        assertThat(error.getMessage(), containsString("its mapping must have [index] set to [true]"));

        var esqlContext = new EsqlSearchExecutionContext(context, null);
        Query expected = new DenseVectorQueryBuilder("vector", VECTOR, null, false).toQuery(esqlContext);
        if (threshold != null) {
            expected = new VectorSimilarityQuery(expected, threshold, (1 + threshold) / 2);
        }
        assertEquals(expected, query.toQuery(esqlContext));
    }

    public void testPrefilter() throws IOException {
        var context = new EsqlSearchExecutionContext(createSearchExecutionContext(mapper()), null);
        var filter = termQuery("id", 2);
        var query = new KnnVectorQueryBuilder("vector", VECTOR, 1, 10, null, null, null).addFilterQuery(filter);
        var expected = new BooleanQuery.Builder().add(
            new DenseVectorQueryBuilder("vector", VECTOR, null, false).toQuery(context),
            BooleanClause.Occur.MUST
        )
            .add(new BooleanQuery.Builder().add(filter.toQuery(context), BooleanClause.Occur.FILTER).build(), BooleanClause.Occur.FILTER)
            .build();
        assertEquals(expected, query.toQuery(context));
    }

    private MapperService mapper() throws IOException {
        return createMapperService(mapping(b -> {
            b.startObject("vector").field("type", "dense_vector").field("dims", 3).field("index", false).endObject();
            b.startObject("id").field("type", "integer").endObject();
        }));
    }
}
