package com.vectordb;

import java.util.*;

public class DocumentDB {
    public static class DocItem {
        public int id;
        public String title;
        public String text;
        public List<Float> emb;

        public DocItem(int id, String title, String text, List<Float> emb) {
            this.id = id;
            this.title = title;
            this.text = text;
            this.emb = emb;
        }
    }

    private final Map<Integer, DocItem> store = new HashMap<>();
    private HNSW hnsw;
    private final BruteForce bf = new BruteForce();
    private int nextId = 1;
    private int dims = 0;

    public synchronized int insert(String title, String text, List<Float> emb) {
        if (dims == 0) {
            dims = emb.size();
            hnsw = new HNSW(16, 200);
        }
        DocItem item = new DocItem(nextId++, title, text, emb);
        store.put(item.id, item);
        
        VectorItem vi = new VectorItem(item.id, title, "", emb);
        bf.insert(vi);
        hnsw.insert(vi, DistanceMetric::cosine);
        return item.id;
    }

    public synchronized boolean remove(int id) {
        if (!store.containsKey(id)) return false;
        store.remove(id);
        bf.remove(id);
        hnsw.remove(id);
        return true;
    }

    public synchronized List<DocItem> all() {
        return new ArrayList<>(store.values());
    }

    public synchronized List<Map<String, Object>> search(List<Float> q, int k) {
        if (hnsw == null) return new ArrayList<>();
        
        List<BruteForce.SearchResult> results;
        if (store.size() < 10) {
            results = bf.knn(q, k, DistanceMetric::cosine);
        } else {
            results = hnsw.knn(q, k, 50, DistanceMetric::cosine);
        }

        List<Map<String, Object>> hits = new ArrayList<>();
        for (BruteForce.SearchResult r : results) {
            DocItem item = store.get(r.id);
            if (item != null) {
                Map<String, Object> hit = new HashMap<>();
                hit.put("id", item.id);
                hit.put("title", item.title);
                hit.put("text", item.text);
                hit.put("distance", r.distance);
                hits.add(hit);
            }
        }
        return hits;
    }

    public synchronized int getDims() {
        return dims;
    }
}
