# SmartRAG: Intelligent Retrieval-Augmented Generation

## 🧠 [Workflow](https://claude.ai/public/artifacts/af9c4502-17d8-4300-9cef-e03dbcfb86fd)

1. **User Query Input**
2. Retrieve documents from internal knowledge base (LlamaIndex)
3. Evaluate:
   - Relevance of passages (GPT)
   - Time sensitivity of query (GPT)
4. Decide:
   - Use internal results
   - Or make external request (Perplexity)
5. Return response with source and metadata

![overflow](https://linkingblog.oss-eu-central-1.aliyuncs.com/picgo/20250508011406.png)

**SmartRAG** is an intelligent RAG (Retrieval-Augmented Generation) system that dynamically decides whether to use internal knowledge base results or make external network requests. This decision is based on **query relevance**, **retrieval quality**, and **time sensitivity** — ensuring efficient, context-aware, and up-to-date responses.

## SmartRAG API: Intelligent Retrieval-Augmented Generation

A **Retrieval-Augmented Generation (RAG)** system that dynamically decides whether to serve answers from an internal **LlamaIndex** knowledge base or fetch up-to-date information via the **Perplexity API**, based on query relevance and time sensitivity.

### 🛠️ Features

- **Hybrid Retrieval**: Combined dense & sparse search using LlamaIndex
- **Relevance Scoring**: GPT-4 evaluates how well retrieved passages address the query
- **Time Sensitivity Detection**: GPT-4 assesses if a query requires recent information
- **Decision Engine**: Prompt-driven logic decides between internal vs external sources
- **External Fetch**: Real-time data via Perplexity API when internal coverage is insufficient
- **Streaming & Non-Streaming Endpoints**: FastAPI-powered API with both batched and SSE streams
- **CORS Enabled**: Configurable origins for front-end integration
- **Structured Logging**: Optional logging of decision metadata for debugging and analysis

### 📦 Installation

1. **Clone the repo**

   ```
   git clone https://github.com/your-org/SmartRAG.git
   cd SmartRAG/Code/Py/{your_path}
   ```

2. **Create a virtual environment & install dependencies**

   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables** Create a `.env` file in the project root with the following keys:

   ```
   LLAMA_INDEX_ORG_ID=your_llama_org_id
   LLAMA_INDEX_KEY=your_llama_api_key
   OPENAI_API_KEY=your_openai_key
   PPLX_API_KEY=your_perplexity_key
   ```

### 🚀 Running the API

Start the FastAPI server with **uvicorn**:

```
uvicorn onlineRagApi:app --host 0.0.0.0 --port 8000 --reload
```

By default, the API listens on port `8000`.

### 🔌 API Endpoints

#### 1. Health Check

```
GET /
```

- **Description**: Verifies that the SmartRAG API is running

- **Response**:

  ```
  {
    "message": "SmartRAG API is running",
    "docs": "/docs"
  }
  ```

#### 2. Non-Streaming Query

```
POST /query
```

- **Description**: Processes a query end-to-end and returns the full response and metadata.

- **Request Body**:

  ```
  { "query": "Your question here" }
  ```

- **Response**:

  ```
  {
    "query": "...",
    "source": "internal" | "external",
    "response": "...",
    "citations": [ ... ],        // only for external
    "metadata": {
      "relevance_analysis": {...},
      "time_sensitivity_analysis": {...},
      "decision": {...},
      "timestamp": "2025-05-10T12:34:56.789123"
    }
  }
  ```

#### 3. Streaming Query

```
GET /query/stream?query=Your+question
```

- **Description**: Returns intermediate steps (retrieval, relevance, time-sensitivity, decision) as newline-delimited JSON (NDJSON) via `application/x-ndjson`.
- **Usage**: Ideal for real-time UIs that display each reasoning step.

### 🌐 CORS & Front-End Testing

A simple HTML tester is included at `test.html`. It provides buttons for:

- **Non-Streaming** (`/query`)
- **Streaming** (`/query/stream`)

Open `test.html` in your browser (adjust the `fetch` URL if your server runs elsewhere).

### 📡 SSE Demo (Optional)

The `stream.py` FastAPI app shows a standalone **Server-Sent Events** example:

```
# Install SSE dependencies and run:
uvicorn stream:app --reload --host 0.0.0.0 --port 8001
```

Visit `http://localhost:8001/` to see live messages emitted every second.

### 🧪 Testing Script

`TestRag.py` demonstrates initializing the index, running both internal & external queries, and printing nodes, responses, and citations.

```
python TestRag.py
```

### 📝 Example Usage

```
# Non-Streaming
curl -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{"query":"What's the latest on Denmark's energy policy?"}'

# Streaming
curl http://localhost:8000/query/stream?query="Denmark%20energy%20policy"
```

### 🔍 Workflow Diagram



*Version 1.0.0*
