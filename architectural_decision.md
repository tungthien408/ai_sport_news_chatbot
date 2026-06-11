# AI Sport News Chatbot: Architectural Review

## Executive Summary
This document provides an architectural review of the AI Sport News Chatbot project. The architecture centers around a Retrieval-Augmented Generation (RAG) pipeline built using LangGraph, PostgreSQL (PGVector), and external LLM endpoints. Its primary goal is to answer user queries with accurate, up-to-date sports news while maintaining conversational context.

---

## Core Architectural Decisions

### 1. LangGraph as the Core Orchestrator
**Technologies Used:** `langgraph`, `langchain-core`

The chatbot relies on a compiled `StateGraph` from LangGraph rather than the standard LangChain `AgentExecutor`. The workflow acts as a state machine that classifies queries (`sports_news`, `sports_wiki`, `off_topic`), performs recency checks on retrieved documents, and conditionally triggers external fallbacks.

**Pros:**
- **Deterministic Control Flow:** Offers precise control over the conversation loop. By explicitly defining nodes and edges, it eliminates the unpredictability of autonomous AgentExecutors choosing the wrong tool.
- **Clear Routing:** Conditional edges cleanly separate distinct pathways (e.g., handling recent news RAG vs. Wikipedia knowledge lookups vs. polite off-topic rejections).
- **Built-in Memory Management:** Passes message history natively within the `GraphState` (using LangGraph's reducer capabilities) without complex memory buffer chains.

**Cons:**
- **Increased Boilerplate:** Requires explicitly defining nodes, edges, and state schemas (`GraphState`), which increases the setup time compared to simple, out-of-the-box agent constructors.
- **Reduced Autonomy:** The LLM does not dynamically choose multi-step tool sequences on the fly; it must strictly follow the defined graph edges.

**P/s**: in companies, they do avoid using langchain + langgraph as it does not guarantee some key factors such as latency, scalability, cost, security and etc.

### 2. PostgreSQL + PGVector for the Vector Store
**Technologies Used:** `langchain-postgres`, `psycopg2-binary`, `sqlalchemy`, PostgreSQL, PGVector extension, Docker

Instead of using a managed vector database SaaS (like Pinecone) or a local flat-file store (like ChromaDB), the project utilizes a containerized PostgreSQL instance loaded with the PGVector extension.

**Pros:**
- **Unified Datastore:** Capable of storing both structured relational data (e.g., planned user session persistence) and vector embeddings within the exact same database infrastructure.
- **Cost-Effective & Local:** Free to run locally via Docker (`docker-compose`), avoiding cloud vector database quotas and pricing limits during development.
- **Automatic Upserts & Deduplication:** By generating deterministic UUIDs (hashing the URLs and chunk indices), PGVector handles upserts automatically, preventing duplicate content insertion.

**Cons:**
- **Operational Overhead:** Requires manual Docker management, volume persistence strategies, and database initializations compared to serverless/cloud-managed alternatives.
- **Scaling Complexity:** Scaling PostgreSQL for high-dimensional vector search at a massive scale requires more DBA expertise than using specialized distributed vector DBs.

### 3. LLM Models and Embeddings (NVIDIA & QWEN)
**Technologies Used:** `langchain-nvidia-ai-endpoints`, NVIDIA API (`nemotron-3-super-120b-a12b`), `nvidia/nv-embed-v1`

The project leverages high-performance endpoints from NVIDIA for both embedding generation (with a batch size of 16) and primary LLM generation/classification tasks. 

**Pros:**
- **High Quality & Fast Inference:** The `nemotron-3-super-120b` model provides highly accurate reasoning capabilities, which excels in zero-shot classification (the gatekeeper node) and context-based summarization.
- **Decoupled Architecture:** Using standard `langchain` wrappers makes swapping LLMs (e.g., switching to QWEN, as mentioned in `.env`) trivial by simply changing the model ID and API keys.

**Cons:**
- **Vendor Lock-in Risk:** Relying strictly on external API endpoints creates a dependency. If the external service experiences downtime, the chatbot ceases to function.
- **Latency Overheads:** External API calls introduce network latency, which stacks up in a multi-step graph execution (e.g., 1 call for classification + 1 call for answer generation).

### 4. Wikipedia API as a Fallback Knowledge Base
**Technologies Used:** `wikipedia` Python package

If the vector database yields no recent news (`NOT_FOUND`), or the user explicitly asks a general knowledge sports question (`sports_wiki`), the graph routes to search Wikipedia dynamically.

**Pros:**
- **Mitigates Hallucinations:** Provides the LLM with a highly reliable, encyclopedic grounding source when the proprietary crawled news dataset lacks the answer.
- **Multilingual Support:** The custom `WikipediaSearcher` implements Regex-based detection to search in Vietnamese (`vi`) first, falling back to English, improving localization.

**Cons:**
- **Variable Data Quality:** Wikipedia summaries vary significantly in length and detail. Disambiguation errors can sometimes lead to irrelevant context if the heuristic fallback fails.
- **Rate Limiting:** Heavy concurrent usage of the standard `wikipedia` library (which scrapes Wikipedia's API) could lead to rate limits or temporary IP blocks.

### 5. OOP-based Graph Node Encapsulation
**Technologies Used:** Python Classes (`GraphNodeManager`, `WikipediaSearcher`)

Nodes are implemented as methods inside a `GraphNodeManager` class, rather than a collection of scattered, standalone functions.

**Pros:**
- **Clean Dependency Injection:** Easy to initialize and share the LLM instance and searchers across multiple nodes without relying on global variables.
- **High Readability:** Encapsulating system prompts, node handlers, and routing logic into one cohesive manager object makes the graph builder (`graph.py`) exceptionally clean and maintainable.

**Cons:**
- **State Mutation Risks:** Since nodes are methods of an instance, developers must be extremely careful not to accidentally store mutable, request-level state on `self`, ensuring the instance remains completely stateless and thread-safe.

### 6. Crawling & Data Ingestion Pipeline
**Technologies Used:** `apscheduler`, `feedparser`, `beautifulsoup4`, `trafilatura`

The backend includes a crawler pipeline that parses RSS feeds, extracts HTML text, cleans it, chunks it, and upserts it into the vector database.

**Pros:**
- **Robust Text Extraction:** `trafilatura` is highly effective at extracting the main body text from complex news articles while automatically stripping away boilerplate, navigation menus, and ads.
- **Automated Freshness:** `apscheduler` allows for continuous, hands-free updates of the database to keep the chatbot current.

**Cons:**
- **DOM Fragility:** Scraping libraries like `beautifulsoup4` can break immediately if the target news websites change their CSS classes or HTML structure.
- **Compute Intensive:** Continuous background crawling, cleaning, and re-embedding can be CPU and API-heavy if delta-checks are not optimized.

### 7. Interface: CLI (Current) vs. FastAPI/React (Planned Phase 3 & 4)
**Technologies Used:** Python `input()` loop (Current) -> `FastAPI`, `Pydantic`, `uvicorn`, `React.js` (Planned)

The current interface is a terminal loop (`chat.py`). Project plans dictate moving to a robust FastAPI backend with a React frontend.

**Pros (Planned Architecture):**
- **Scalability & Concurrency:** FastAPI allows asynchronous concurrent requests, making the bot accessible to multiple users at once.
- **Strict Data Contracts:** Pydantic will ensure strict input/output validation, preventing malformed requests from breaking the graph.
- **Persistence:** Allows for real session memory using PostgreSQL/SQLAlchemy, keeping conversation history alive across browser sessions.

**Cons (Current CLI):**
- **Single-User Bottleneck:** The CLI blocks on `input()`, meaning it can only handle one conversational turn at a time.
- **Ephemeral State:** Memory is completely lost when the terminal process exits.
