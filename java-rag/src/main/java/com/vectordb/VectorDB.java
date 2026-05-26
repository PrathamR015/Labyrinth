package com.vectordb;

import java.util.*;

public class VectorDB {
    private final Map<Integer, VectorItem> store = new HashMap<>();
    private final BruteForce bf = new BruteForce();
    private final KDTree kdt;
    private final HNSW hnsw;
    private int nextId = 1;
    public final int dims;

    public VectorDB(int d) {
        this.dims = d;
        this.kdt = new KDTree(d);
        this.hnsw = new HNSW(16, 200);
    }

    public synchronized int insert(String meta, String cat, List<Float> emb, DistanceMetric dist) {
        VectorItem v = new VectorItem(nextId++, meta, cat, emb);
        store.put(v.id, v);
        bf.insert(v);
        kdt.insert(v);
        hnsw.insert(v, dist);
        return v.id;
    }

    public synchronized boolean remove(int id) {
        if (!store.containsKey(id)) return false;
        store.remove(id);
        bf.remove(id);
        hnsw.remove(id);
        kdt.rebuild(new ArrayList<>(store.values()));
        return true;
    }

    public static class Hit {
        public int id;
        public String metadata, category;
        public List<Float> embedding;
        public float distance;

        public Hit(VectorItem item, float distance) {
            this.id = item.id;
            this.metadata = item.metadata;
            this.category = item.category;
            this.embedding = item.emb;
            this.distance = distance;
        }
    }

    public static class SearchOut {
        public List<Hit> hits = new ArrayList<>();
        public long latencyUs;
        public String algo, metric;
    }

    public synchronized SearchOut search(List<Float> q, int k, String metricName, String algo) {
        DistanceMetric dist = DistanceMetric.get(metricName);
        long t0 = System.nanoTime();

        List<BruteForce.SearchResult> raw;
        if ("bruteforce".equals(algo)) {
            raw = bf.knn(q, k, dist);
        } else if ("kdtree".equals(algo)) {
            raw = kdt.knn(q, k, dist);
        } else {
            raw = hnsw.knn(q, k, 50, dist);
        }

        long us = (System.nanoTime() - t0) / 1000;

        SearchOut out = new SearchOut();
        out.latencyUs = us;
        out.algo = algo;
        out.metric = metricName;
        for (BruteForce.SearchResult r : raw) {
            if (store.containsKey(r.id)) {
                out.hits.add(new Hit(store.get(r.id), r.distance));
            }
        }
        return out;
    }

    public synchronized Map<String, Object> benchmark(List<Float> q, int k, String metricName) {
        DistanceMetric dist = DistanceMetric.get(metricName);
        
        long tBf = System.nanoTime();
        bf.knn(q, k, dist);
        long bfUs = (System.nanoTime() - tBf) / 1000;

        long tKd = System.nanoTime();
        kdt.knn(q, k, dist);
        long kdUs = (System.nanoTime() - tKd) / 1000;

        long tHnsw = System.nanoTime();
        hnsw.knn(q, k, 50, dist);
        long hnswUs = (System.nanoTime() - tHnsw) / 1000;

        Map<String, Object> res = new HashMap<>();
        res.put("bfUs", bfUs);
        res.put("kdUs", kdUs);
        res.put("hnswUs", hnswUs);
        res.put("n", store.size());
        return res;
    }

    public synchronized List<VectorItem> all() {
        return new ArrayList<>(store.values());
    }

    public synchronized Map<String, Object> hnswInfo() {
        return hnsw.getInfo();
    }

    public synchronized int size() {
        return store.size();
    }
}
