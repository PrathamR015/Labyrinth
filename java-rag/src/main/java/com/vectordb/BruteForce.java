package com.vectordb;

import java.util.*;

public class BruteForce {
    private final List<VectorItem> items = new ArrayList<>();

    public synchronized void insert(VectorItem v) {
        items.add(v);
    }

    public synchronized List<SearchResult> knn(List<Float> q, int k, DistanceMetric dist) {
        List<SearchResult> results = new ArrayList<>();
        for (VectorItem item : items) {
            results.add(new SearchResult(dist.calculate(q, item.emb), item.id));
        }
        results.sort(Comparator.comparingDouble(r -> r.distance));
        if (results.size() > k) {
            return results.subList(0, k);
        }
        return results;
    }

    public synchronized void remove(int id) {
        items.removeIf(v -> v.id == id);
    }

    public static class SearchResult {
        public float distance;
        public int id;

        public SearchResult(float distance, int id) {
            this.distance = distance;
            this.id = id;
        }
    }
}
