package com.vectordb;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class HNSW {
    private static class Node {
        VectorItem item;
        int maxLyr;
        List<List<Integer>> nbrs;

        Node(VectorItem item, int maxLyr) {
            this.item = item;
            this.maxLyr = maxLyr;
            this.nbrs = new ArrayList<>();
            for (int i = 0; i <= maxLyr; i++) {
                nbrs.add(new ArrayList<>());
            }
        }
    }

    private final Map<Integer, Node> nodes = new ConcurrentHashMap<>();
    private final int M, M0, ef_build;
    private final float mL;
    private int topLayer = -1;
    private int entryPt = -1;
    private final Random rng = new Random(42);

    public HNSW(int m, int efBuild) {
        this.M = m;
        this.M0 = 2 * m;
        this.ef_build = efBuild;
        this.mL = (float) (1.0 / Math.log(m));
    }

    private int randLevel() {
        return (int) Math.floor(-Math.log(rng.nextFloat()) * mL);
    }

    private List<BruteForce.SearchResult> searchLayer(List<Float> q, int ep, int ef, int lyr, DistanceMetric dist) {
        Set<Integer> vis = new HashSet<>();
        PriorityQueue<BruteForce.SearchResult> cands = new PriorityQueue<>(Comparator.comparingDouble(r -> r.distance));
        PriorityQueue<BruteForce.SearchResult> found = new PriorityQueue<>((a, b) -> Float.compare(b.distance, a.distance));

        float d0 = dist.calculate(q, nodes.get(ep).item.emb);
        vis.add(ep);
        cands.add(new BruteForce.SearchResult(d0, ep));
        found.add(new BruteForce.SearchResult(d0, ep));

        while (!cands.isEmpty()) {
            BruteForce.SearchResult curr = cands.poll();
            if (found.size() >= ef && curr.distance > found.peek().distance) break;
            
            Node currNode = nodes.get(curr.id);
            if (lyr >= currNode.nbrs.size()) continue;
            
            for (int nid : currNode.nbrs.get(lyr)) {
                if (vis.contains(nid) || !nodes.containsKey(nid)) continue;
                vis.add(nid);
                float nd = dist.calculate(q, nodes.get(nid).item.emb);
                if (found.size() < ef || nd < found.peek().distance) {
                    cands.add(new BruteForce.SearchResult(nd, nid));
                    found.add(new BruteForce.SearchResult(nd, nid));
                    if (found.size() > ef) found.poll();
                }
            }
        }

        List<BruteForce.SearchResult> res = new ArrayList<>(found);
        res.sort(Comparator.comparingDouble(r -> r.distance));
        return res;
    }

    private List<Integer> selectNbrs(List<BruteForce.SearchResult> cands, int maxM) {
        List<Integer> r = new ArrayList<>();
        for (int i = 0; i < Math.min(cands.size(), maxM); i++) {
            r.add(cands.get(i).id);
        }
        return r;
    }

    public synchronized void insert(VectorItem item, DistanceMetric dist) {
        int id = item.id;
        int lvl = randLevel();
        Node newNode = new Node(item, lvl);
        nodes.put(id, newNode);

        if (entryPt == -1) {
            entryPt = id;
            topLayer = lvl;
            return;
        }

        int ep = entryPt;
        for (int lc = topLayer; lc > lvl; lc--) {
            List<BruteForce.SearchResult> W = searchLayer(item.emb, ep, 1, lc, dist);
            if (!W.isEmpty()) ep = W.get(0).id;
        }

        for (int lc = Math.min(topLayer, lvl); lc >= 0; lc--) {
            List<BruteForce.SearchResult> W = searchLayer(item.emb, ep, ef_build, lc, dist);
            int maxM = (lc == 0) ? M0 : M;
            List<Integer> sel = selectNbrs(W, maxM);
            newNode.nbrs.get(lc).addAll(sel);

            for (int nid : sel) {
                Node neighbor = nodes.get(nid);
                if (neighbor == null) continue;
                if (neighbor.nbrs.size() <= lc) {
                    while (neighbor.nbrs.size() <= lc) neighbor.nbrs.add(new ArrayList<>());
                }
                List<Integer> conn = neighbor.nbrs.get(lc);
                conn.add(id);
                if (conn.size() > maxM) {
                    List<BruteForce.SearchResult> ds = new ArrayList<>();
                    for (int c : conn) {
                        if (nodes.containsKey(c)) {
                            ds.add(new BruteForce.SearchResult(dist.calculate(neighbor.item.emb, nodes.get(c).item.emb), c));
                        }
                    }
                    ds.sort(Comparator.comparingDouble(r -> r.distance));
                    conn.clear();
                    for (int i = 0; i < maxM && i < ds.size(); i++) {
                        conn.add(ds.get(i).id);
                    }
                }
            }
            if (!W.isEmpty()) ep = W.get(0).id;
        }

        if (lvl > topLayer) {
            topLayer = lvl;
            entryPt = id;
        }
    }

    public List<BruteForce.SearchResult> knn(List<Float> q, int k, int ef, DistanceMetric dist) {
        if (entryPt == -1) return new ArrayList<>();
        int ep = entryPt;
        for (int lc = topLayer; lc > 0; lc--) {
            List<BruteForce.SearchResult> W = searchLayer(q, ep, 1, lc, dist);
            if (!W.isEmpty()) ep = W.get(0).id;
        }
        List<BruteForce.SearchResult> W = searchLayer(q, ep, Math.max(ef, k), 0, dist);
        if (W.size() > k) return W.subList(0, k);
        return W;
    }

    public synchronized void remove(int id) {
        if (!nodes.containsKey(id)) return;
        for (Node nd : nodes.values()) {
            for (List<Integer> layer : nd.nbrs) {
                layer.removeIf(nid -> nid == id);
            }
        }
        if (entryPt == id) {
            entryPt = -1;
            for (int nid : nodes.keySet()) {
                if (nid != id) {
                    entryPt = nid;
                    break;
                }
            }
        }
        nodes.remove(id);
    }

    public int size() {
        return nodes.size();
    }

    // For HNSW Info endpoint
    public Map<String, Object> getInfo() {
        Map<String, Object> gi = new HashMap<>();
        gi.put("topLayer", topLayer);
        gi.put("nodeCount", nodes.size());
        
        List<Integer> nodesPerLayer = new ArrayList<>();
        List<Integer> edgesPerLayer = new ArrayList<>();
        int maxL = Math.max(topLayer + 1, 1);
        for (int i = 0; i < maxL; i++) {
            nodesPerLayer.add(0);
            edgesPerLayer.add(0);
        }

        List<Map<String, Object>> nodesList = new ArrayList<>();
        List<Map<String, Object>> edgesList = new ArrayList<>();

        for (Node nd : nodes.values()) {
            Map<String, Object> nm = new HashMap<>();
            nm.put("id", nd.item.id);
            nm.put("metadata", nd.item.metadata);
            nm.put("category", nd.item.category);
            nm.put("maxLyr", nd.maxLyr);
            nodesList.add(nm);

            for (int lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
                nodesPerLayer.set(lc, nodesPerLayer.get(lc) + 1);
                if (lc < nd.nbrs.size()) {
                    for (int nid : nd.nbrs.get(lc)) {
                        if (nd.item.id < nid) {
                            edgesPerLayer.set(lc, edgesPerLayer.get(lc) + 1);
                            Map<String, Object> em = new HashMap<>();
                            em.put("src", nd.item.id);
                            em.put("dst", nid);
                            em.put("lyr", lc);
                            edgesList.add(em);
                        }
                    }
                }
            }
        }
        gi.put("nodesPerLayer", nodesPerLayer);
        gi.put("edgesPerLayer", edgesPerLayer);
        gi.put("nodes", nodesList);
        gi.put("edges", edgesList);
        return gi;
    }
}
