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

------

## 🔧 Features

- **Hybrid Retrieval** using [LlamaIndex](https://llamaindex.ai/) with dense and sparse vector search
- **Relevance Scoring** via GPT (OpenAI `gpt-4`)
- **Time Sensitivity Detection** to identify queries requiring recent updates
- **Decision Engine** powered by prompt-driven logic to choose between internal and external sources
- **External Retrieval** via [Perplexity API](https://www.perplexity.ai/) for current events and real-time information
- **Structured Logging** for decision transparency and system debugging

------

## 🚀 Example

```bash
python smart_rag.py
```

Example query:

```bash
"Trump's 2025 tariff policy impact on Denmark"
```

The system analyzes the query and either:

- Uses internal knowledge base if relevant and recent enough
- Or fetches real-time insights from Perplexity if time-sensitive or not well covered
