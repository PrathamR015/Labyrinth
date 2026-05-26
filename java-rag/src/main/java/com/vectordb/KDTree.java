package com.vectordb;

import java.util.*;

public class KDTree {
    private static class KDNode {
        VectorItem item;
        KDNode left, right;

        KDNode(VectorItem item) {
            this.item = item;
        }
    }

    private KDNode root;
    private final int dims;

    public KDTree(int dims) {
        this.dims = dims;
    }

    public synchronized void insert(VectorItem v) {
        root = insert(root, v, 0);
    }

    private KDNode insert(KDNode n, VectorItem v, int d) {
        if (n == null) return new KDNode(v);
        int ax = d % dims;
        if (v.emb.get(ax) < n.item.emb.get(ax)) {
            n.left = insert(n.left, v, d + 1);
        } else {
            n.right = insert(n.right, v, d + 1);
        }
        return n;
    }

    public synchronized List<BruteForce.SearchResult> knn(List<Float> q, int k, DistanceMetric dist) {
        PriorityQueue<BruteForce.SearchResult> heap = new PriorityQueue<>((a, b) -> Float.compare(b.distance, a.distance));
        knn(root, q, k, 0, dist, heap);
        List<BruteForce.SearchResult> results = new ArrayList<>();
        while (!heap.isEmpty()) {
            results.add(heap.poll());
        }
        results.sort(Comparator.comparingDouble(r -> r.distance));
        return results;
    }

    private void knn(KDNode n, List<Float> q, int k, int d, DistanceMetric dist, PriorityQueue<BruteForce.SearchResult> heap) {
        if (n == null) return;
        float dn = dist.calculate(q, n.item.emb);
        if (heap.size() < k || dn < heap.peek().distance) {
            heap.add(new BruteForce.SearchResult(dn, n.item.id));
            if (heap.size() > k) heap.poll();
        }
        int ax = d % dims;
        float diff = q.get(ax) - n.item.emb.get(ax);
        KDNode closer = diff < 0 ? n.left : n.right;
        KDNode farther = diff < 0 ? n.right : n.left;
        knn(closer, q, k, d + 1, dist, heap);
        if (heap.size() < k || Math.abs(diff) < heap.peek().distance) {
            knn(farther, q, k, d + 1, dist, heap);
        }
    }

    public synchronized void rebuild(List<VectorItem> items) {
        root = null;
        for (VectorItem v : items) insert(v);
    }
}
