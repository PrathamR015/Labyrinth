package com.vectordb;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import spark.Spark;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

import static spark.Spark.*;

public class Main {
    private static final int DIMS = 16;
    private static final Gson gson = new Gson();

    public static void main(String[] args) {
        VectorDB db = new VectorDB(DIMS);
        DocumentDB docDB = new DocumentDB();
        OllamaClient ollama = new OllamaClient("127.0.0.1", 11434);

        loadDemo(db);

        boolean ollamaUp = ollama.isAvailable();
        System.out.println("=== Labyrinth Engine (Java) ===");
        System.out.println("http://localhost:8080");
        System.out.println(db.size() + " demo vectors | " + DIMS + " dims | HNSW+KD-Tree+BruteForce");
        System.out.println("Ollama: " + (ollamaUp ? "ONLINE" : "OFFLINE (install from ollama.com)"));
        if (ollamaUp) {
            System.out.println("  embed model: " + ollama.embedModel + "  gen model: " + ollama.genModel);
        }

        port(8080);

        // CORS
        options("/*", (request, response) -> {
            String accessControlRequestHeaders = request.headers("Access-Control-Request-Headers");
            if (accessControlRequestHeaders != null) {
                response.header("Access-Control-Allow-Headers", accessControlRequestHeaders);
            }
            String accessControlRequestMethod = request.headers("Access-Control-Request-Method");
            if (accessControlRequestMethod != null) {
                response.header("Access-Control-Allow-Methods", accessControlRequestMethod);
            }
            return "OK";
        });

        before((request, response) -> {
            response.header("Access-Control-Allow-Origin", "*");
            response.header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
            response.header("Access-Control-Allow-Headers", "Content-Type");
        });

        // Demo Endpoints
        get("/search", (req, res) -> {
            List<Float> q = parseVec(req.queryParams("v"));
            if (q.size() != DIMS) {
                res.status(400);
                return "{\"error\":\"need " + DIMS + "D vector\"}";
            }
            int k = Integer.parseInt(req.queryParamOrDefault("k", "5"));
            String metric = req.queryParamOrDefault("metric", "cosine");
            String algo = req.queryParamOrDefault("algo", "hnsw");

            VectorDB.SearchOut out = db.search(q, k, metric, algo);
            res.type("application/json");
            return gson.toJson(out);
        });

        post("/insert", (req, res) -> {
            JsonObject body = gson.fromJson(req.body(), JsonObject.class);
            String meta = body.get("metadata").getAsString();
            String cat = body.get("category").getAsString();
            List<Float> emb = parseVec(body.get("embedding").getAsString());
            int id = db.insert(meta, cat, emb, DistanceMetric.get("cosine"));
            res.type("application/json");
            return "{\"id\":" + id + "}";
        });

        delete("/delete/:id", (req, res) -> {
            int id = Integer.parseInt(req.params(":id"));
            boolean ok = db.remove(id);
            res.type("application/json");
            return "{\"ok\":" + ok + "}";
        });

        get("/items", (req, res) -> {
            res.type("application/json");
            return gson.toJson(db.all());
        });

        get("/benchmark", (req, res) -> {
            List<Float> q = parseVec(req.queryParams("v"));
            int k = Integer.parseInt(req.queryParamOrDefault("k", "5"));
            String metric = req.queryParamOrDefault("metric", "cosine");
            res.type("application/json");
            return gson.toJson(db.benchmark(q, k, metric));
        });

        get("/hnsw-info", (req, res) -> {
            res.type("application/json");
            return gson.toJson(db.hnswInfo());
        });

        get("/stats", (req, res) -> {
            Map<String, Object> stats = new HashMap<>();
            stats.put("count", db.size());
            stats.put("dims", DIMS);
            stats.put("algorithms", List.of("bruteforce", "kdtree", "hnsw"));
            stats.put("metrics", List.of("euclidean", "cosine", "manhattan"));
            res.type("application/json");
            return gson.toJson(stats);
        });

        // Document Endpoints
        post("/doc/insert", (req, res) -> {
            JsonObject body = gson.fromJson(req.body(), JsonObject.class);
            String title = body.get("title").getAsString();
            String text = body.get("text").getAsString();
            
            List<String> chunks = chunkText(text, 250, 30);
            List<Integer> ids = new ArrayList<>();

            for (int i = 0; i < chunks.size(); i++) {
                List<Float> emb = ollama.embed(chunks.get(i));
                if (emb.isEmpty()) {
                    res.status(503);
                    return "{\"error\":\"Ollama unavailable\"}";
                }
                String chunkTitle = chunks.size() > 1 ? title + " [" + (i + 1) + "/" + chunks.size() + "]" : title;
                ids.add(docDB.insert(chunkTitle, chunks.get(i), emb));
            }

            Map<String, Object> out = new HashMap<>();
            out.put("ids", ids);
            out.put("chunks", chunks.size());
            out.put("dims", docDB.getDims());
            res.type("application/json");
            return gson.toJson(out);
        });

        get("/doc/list", (req, res) -> {
            List<DocumentDB.DocItem> docs = docDB.all();
            List<Map<String, Object>> out = docs.stream().map(d -> {
                Map<String, Object> m = new HashMap<>();
                m.put("id", d.id);
                m.put("title", d.title);
                String preview = d.text.substring(0, Math.min(d.text.length(), 120));
                if (d.text.length() > 120) preview += "…";
                m.put("preview", preview);
                m.put("words", d.text.split("\\s+").length);
                return m;
            }).collect(Collectors.toList());
            res.type("application/json");
            return gson.toJson(out);
        });

        delete("/doc/delete/:id", (req, res) -> {
            int id = Integer.parseInt(req.params(":id"));
            boolean ok = docDB.remove(id);
            res.type("application/json");
            return "{\"ok\":" + ok + "}";
        });

        post("/doc/search", (req, res) -> {
            JsonObject body = gson.fromJson(req.body(), JsonObject.class);
            String question = body.get("question").getAsString();
            int k = body.has("k") ? body.get("k").getAsInt() : 3;

            List<Float> qEmb = ollama.embed(question);
            if (qEmb.isEmpty()) {
                res.status(503);
                return "{\"error\":\"Ollama unavailable\"}";
            }

            List<Map<String, Object>> hits = docDB.search(qEmb, k);
            Map<String, Object> out = new HashMap<>();
            out.put("contexts", hits);
            res.type("application/json");
            return gson.toJson(out);
        });

        post("/doc/ask", (req, res) -> {
            JsonObject body = gson.fromJson(req.body(), JsonObject.class);
            String question = body.get("question").getAsString();
            int k = body.has("k") ? body.get("k").getAsInt() : 3;

            List<Float> qEmb = ollama.embed(question);
            if (qEmb.isEmpty()) {
                res.status(503);
                return "{\"error\":\"Ollama unavailable\"}";
            }

            List<Map<String, Object>> hits = docDB.search(qEmb, k);
            StringBuilder context = new StringBuilder();
            for (Map<String, Object> h : hits) {
                context.append(h.get("text")).append("\n\n");
            }

            String prompt = "You are a helpful assistant. Use the following context to answer the question. If the answer is not in the context, say you don't know.\n\n"
                    + "CONTEXT:\n" + context + "\n\n"
                    + "QUESTION: " + question + "\n\n"
                    + "ANSWER:";

            String answer = ollama.generate(prompt);

            Map<String, Object> out = new HashMap<>();
            out.put("answer", answer);
            out.put("contexts", hits);
            res.type("application/json");
            return gson.toJson(out);
        });

        get("/status", (req, res) -> {
            Map<String, Object> s = new HashMap<>();
            s.put("ollama", ollama.isAvailable() ? "ONLINE" : "OFFLINE");
            s.put("embedModel", ollama.embedModel);
            s.put("genModel", ollama.genModel);
            res.type("application/json");
            return gson.toJson(s);
        });

        // Serve index.html
        get("/", (req, res) -> {
            try {
                return new String(Files.readAllBytes(Paths.get("../index.html")));
            } catch (IOException e) {
                res.status(404);
                return "index.html not found";
            }
        });
    }

    private static void loadDemo(VectorDB db) {
        DistanceMetric dist = DistanceMetric.get("cosine");
        db.insert("Linked List: nodes connected by pointers", "cs", List.of(0.90f,0.85f,0.72f,0.68f,0.12f,0.08f,0.15f,0.10f,0.05f,0.08f,0.06f,0.09f,0.07f,0.11f,0.08f,0.06f), dist);
        db.insert("Binary Search Tree: O(log n) search and insert", "cs", List.of(0.88f,0.82f,0.78f,0.74f,0.15f,0.10f,0.08f,0.12f,0.06f,0.07f,0.08f,0.05f,0.09f,0.06f,0.07f,0.10f), dist);
        db.insert("Dynamic Programming: memoization overlapping subproblems", "cs", List.of(0.82f,0.76f,0.88f,0.80f,0.20f,0.18f,0.12f,0.09f,0.07f,0.06f,0.08f,0.07f,0.08f,0.09f,0.06f,0.07f), dist);
        db.insert("Graph BFS and DFS: breadth and depth first traversal", "cs", List.of(0.85f,0.80f,0.75f,0.82f,0.18f,0.14f,0.10f,0.08f,0.06f,0.09f,0.07f,0.06f,0.10f,0.08f,0.09f,0.07f), dist);
        db.insert("Hash Table: O(1) lookup with collision chaining", "cs", List.of(0.87f,0.78f,0.70f,0.76f,0.13f,0.11f,0.09f,0.14f,0.08f,0.07f,0.06f,0.08f,0.07f,0.10f,0.08f,0.09f), dist);
        
        db.insert("Calculus: derivatives integrals and limits", "math", List.of(0.12f,0.15f,0.18f,0.10f,0.91f,0.86f,0.78f,0.72f,0.08f,0.06f,0.07f,0.09f,0.07f,0.08f,0.06f,0.10f), dist);
        db.insert("Linear Algebra: matrices eigenvalues eigenvectors", "math", List.of(0.20f,0.18f,0.15f,0.12f,0.88f,0.90f,0.82f,0.76f,0.09f,0.07f,0.08f,0.06f,0.10f,0.07f,0.08f,0.09f), dist);
        db.insert("Probability: distributions random variables Bayes theorem", "math", List.of(0.15f,0.12f,0.20f,0.18f,0.84f,0.80f,0.88f,0.82f,0.07f,0.08f,0.06f,0.10f,0.09f,0.06f,0.09f,0.08f), dist);
        db.insert("Number Theory: primes modular arithmetic RSA cryptography", "math", List.of(0.22f,0.16f,0.14f,0.20f,0.80f,0.85f,0.76f,0.90f,0.08f,0.09f,0.07f,0.06f,0.08f,0.10f,0.07f,0.06f), dist);
        db.insert("Combinatorics: permutations combinations generating functions", "math", List.of(0.18f,0.20f,0.16f,0.14f,0.86f,0.78f,0.84f,0.80f,0.06f,0.07f,0.09f,0.08f,0.06f,0.09f,0.10f,0.07f), dist);
        
        db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food", List.of(0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.90f,0.86f,0.78f,0.72f,0.08f,0.06f,0.09f,0.07f), dist);
        db.insert("Sushi: vinegared rice raw fish and nori rolls", "food", List.of(0.06f,0.08f,0.07f,0.09f,0.09f,0.06f,0.08f,0.07f,0.86f,0.90f,0.82f,0.76f,0.07f,0.09f,0.06f,0.08f), dist);
        db.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food", List.of(0.09f,0.07f,0.06f,0.08f,0.08f,0.09f,0.07f,0.06f,0.82f,0.78f,0.90f,0.84f,0.09f,0.07f,0.08f,0.06f), dist);
        db.insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food", List.of(0.07f,0.09f,0.08f,0.06f,0.06f,0.07f,0.09f,0.08f,0.78f,0.82f,0.86f,0.90f,0.06f,0.08f,0.07f,0.09f), dist);
        db.insert("Croissant: laminated pastry with buttery flaky layers", "food", List.of(0.06f,0.07f,0.10f,0.09f,0.10f,0.06f,0.07f,0.10f,0.85f,0.80f,0.76f,0.82f,0.09f,0.07f,0.10f,0.06f), dist);
        
        db.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports", List.of(0.09f,0.07f,0.08f,0.10f,0.08f,0.09f,0.07f,0.06f,0.08f,0.07f,0.09f,0.06f,0.91f,0.85f,0.78f,0.72f), dist);
        db.insert("Football: tackles touchdowns field goals and strategy", "sports", List.of(0.07f,0.09f,0.06f,0.08f,0.09f,0.07f,0.10f,0.08f,0.07f,0.09f,0.08f,0.07f,0.87f,0.89f,0.82f,0.76f), dist);
        db.insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports", List.of(0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.09f,0.06f,0.07f,0.08f,0.83f,0.80f,0.88f,0.82f), dist);
        db.insert("Chess: openings endgames tactics strategic board game", "sports", List.of(0.25f,0.20f,0.22f,0.18f,0.22f,0.18f,0.20f,0.15f,0.06f,0.08f,0.07f,0.09f,0.80f,0.84f,0.78f,0.90f), dist);
        db.insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports", List.of(0.06f,0.08f,0.07f,0.09f,0.08f,0.06f,0.09f,0.07f,0.10f,0.08f,0.06f,0.07f,0.85f,0.82f,0.86f,0.80f), dist);
    }

    private static List<Float> parseVec(String s) {
        if (s == null || s.isEmpty()) return Collections.emptyList();
        try {
            return Arrays.stream(s.split(","))
                    .map(Float::parseFloat)
                    .collect(Collectors.toList());
        } catch (Exception e) {
            return Collections.emptyList();
        }
    }

    private static List<String> chunkText(String text, int chunkWords, int overlapWords) {
        String[] words = text.split("\\s+");
        if (words.length == 0) return Collections.emptyList();
        if (words.length <= chunkWords) return List.of(text);

        List<String> chunks = new ArrayList<>();
        int step = chunkWords - overlapWords;
        for (int i = 0; i < words.length; i += step) {
            int end = Math.min(i + chunkWords, words.length);
            StringBuilder chunk = new StringBuilder();
            for (int j = i; j < end; j++) {
                if (j > i) chunk.append(" ");
                chunk.append(words[j]);
            }
            chunks.add(chunk.toString());
            if (end == words.length) break;
        }
        return chunks;
    }
}
