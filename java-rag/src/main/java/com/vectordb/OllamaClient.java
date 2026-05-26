package com.vectordb;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonArray;
import org.apache.hc.client5.http.classic.methods.HttpPost;
import org.apache.hc.client5.http.classic.methods.HttpGet;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.CloseableHttpResponse;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.apache.hc.core5.http.io.entity.StringEntity;
import org.apache.hc.core5.http.io.entity.EntityUtils;

import java.util.*;

public class OllamaClient {
    private final String host;
    private final int port;
    private final Gson gson = new Gson();

    public String embedModel = "nomic-embed-text";
    public String genModel = "llama3.2";

    public OllamaClient(String host, int port) {
        this.host = host;
        this.port = port;
    }

    public boolean isAvailable() {
        try (CloseableHttpClient httpClient = HttpClients.createDefault()) {
            HttpGet request = new HttpGet("http://" + host + ":" + port + "/api/tags");
            try (CloseableHttpResponse response = httpClient.execute(request)) {
                return response.getCode() == 200;
            }
        } catch (Exception e) {
            return false;
        }
    }

    public List<Float> embed(String text) {
        try (CloseableHttpClient httpClient = HttpClients.createDefault()) {
            HttpPost request = new HttpPost("http://" + host + ":" + port + "/api/embeddings");
            Map<String, String> bodyMap = new HashMap<>();
            bodyMap.put("model", embedModel);
            bodyMap.put("prompt", text);
            request.setEntity(new StringEntity(gson.toJson(bodyMap)));
            request.setHeader("Content-Type", "application/json");

            try (CloseableHttpResponse response = httpClient.execute(request)) {
                if (response.getCode() == 200) {
                    String jsonResponse = EntityUtils.toString(response.getEntity());
                    JsonObject jsonObject = gson.fromJson(jsonResponse, JsonObject.class);
                    JsonArray embeddingArray = jsonObject.getAsJsonArray("embedding");
                    List<Float> embedding = new ArrayList<>();
                    for (int i = 0; i < embeddingArray.size(); i++) {
                        embedding.add(embeddingArray.get(i).getAsFloat());
                    }
                    return embedding;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return Collections.emptyList();
    }

    public String generate(String prompt) {
        try (CloseableHttpClient httpClient = HttpClients.createDefault()) {
            HttpPost request = new HttpPost("http://" + host + ":" + port + "/api/generate");
            Map<String, Object> bodyMap = new HashMap<>();
            bodyMap.put("model", genModel);
            bodyMap.put("prompt", prompt);
            bodyMap.put("stream", false);
            request.setEntity(new StringEntity(gson.toJson(bodyMap)));
            request.setHeader("Content-Type", "application/json");

            try (CloseableHttpResponse response = httpClient.execute(request)) {
                if (response.getCode() == 200) {
                    String jsonResponse = EntityUtils.toString(response.getEntity());
                    JsonObject jsonObject = gson.fromJson(jsonResponse, JsonObject.class);
                    return jsonObject.get("response").getAsString();
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return "ERROR: Ollama unavailable. Run: ollama serve";
    }
}
