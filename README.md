# AI Sport News Chatbot

An LLM-powered chatbot that answers user queries about sports news by crawling, processing, and retrieving information from a vector database.

**Current Status:** Phase 1 in progress — core code and data pipeline scaffolding present, stabilization and conversational memory work underway.

<img width="1031" height="281" alt="image" src="https://github.com/user-attachments/assets/91fcb34c-237a-49df-850a-a70427df724a" />

---

## What exists today

- CLI chat interface: `src/chat.py` (basic interactive CLI)
- RAG orchestration and update logic: `src/rag/rag_update_news.py`
- Crawlers: `src/rag/crawler/news_crawler_v2.py`
- Data pipeline modules: `src/rag/news_pipeline/cleaner.py`, `chunker.py`, `vector_store.py`
- Utilities: `src/util/tools.py`
- Tests: `tests/test_hello.py`
- Docker + postgres compose for a local vector DB (`docker-compose.yml`) and Dockerfile

These components provide the scaffolding for a Retrieval-Augmented Generation (RAG) workflow: crawl -> clean -> chunk -> insert into vector DB -> retrieve -> answer.

---

## True process and progress (up to this point)

The repository implements the basic pipeline components and two crawler versions. The project is actively in development with a focus on improving answer correctness and adding conversational context. Key points:

- Data pipeline: code for crawling, cleaning, chunking, and vector DB operations is present but integration and end-to-end automation are still in development.
- Chat interface: `src/chat.py` works as a CLI entrypoint, but the conversational memory implementation is incomplete (see Known Issues).
- RAG updates: `rag_update_news.py` implements update logic, but automated scheduled ingestion and robust error handling need work.

---

## Known issues and fixes in progress

These were identified during initial testing and are tracked in the codebase and the development plan (see plan.md).

- Wikipedia fallback bug: the current fallback uses `summary()` directly in some places; it should call `search()` first then `summary()` to avoid unexpected failures.
- Prompt isolation and language enforcement: user prompt instructions must be isolated and forced to English to avoid model code-switching.
- `OLD` document warning enforcement: outdated documents need explicit warning markers when surfaced to the model.
- Conversational memory is missing from generation nodes: `chat.py` accumulates `message_history`, but `generate_answer` (in `graph_nodes.py` or the generation logic) builds fresh [SystemMessage, HumanMessage] sequences without injecting recent turns.

---

## Roadmap (Phases)

- **Phase 1: Stabilization & Conversational Memory** (current focus)
  - Implement bug fixes above — in progress.
  - Integrate conversational memory: update generation node to inject the last 3–4 turns into the LLM prompt — planned / in progress.

- **Phase 2: Agentic Reflection & Quantitative Evaluation**
  - Add a reflection "Critic" node after `generate_answer` to detect unsupported claims and loop for rewrites.
  - Build an evaluation harness (20–30 questions) to measure accuracy and hallucination rate before/after reflection.

- **Phase 3: Production Backend & Persistence**
  - Wrap the LangGraph/RAG workflow in FastAPI endpoints with Pydantic models.
  - Persist sessions and histories in PostgreSQL (SQLAlchemy).
  - Update docker-compose to run FastAPI + Postgres + PGVector together.

- **Phase 4: Full-Stack Presentation & Domain Expansion**
  - Build a React-based interactive UI to demonstrate the chatbot.
  - Expand crawlers and sources (RSS feeds, additional sports domains) to show scalability.

You can find a development plan with concrete tasks in `plan.md` at the repository root.

---

## Quick Start

Follow these steps to run the project locally (same as before):

1. Prepare your environment

```bash
cd ./ai_sport_news_chatbot
mkdir -p ./logs
cp .env.example .env  # or create manually (see API Setup)
```

2. Start Docker services

```bash
docker compose up -d db
# Wait for the database to be ready (~10-15 seconds)
```

3. Configure environment variables

Edit the `.env` file and add your API keys and DB URL:

```env
DB_NAME="vectordb"
DB_URL="postgresql+psycopg://langchain:langchain@localhost:5432/vectordb"

NVIDIA_API_KEY=your_actual_key_here
```

4. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

5. Run the application (CLI)

```bash
python ./src/chat.py
```

Note: The current CLI will use the available local vector DB if it has been populated. The end-to-end ingestion automation is still being improved.

---

## API Setup

### Getting API Key(s)

- **NVIDIA_API_KEY:** Get from NVIDIA API Catalog

(Links and details are in the original README and in your internal notes.)

---

## Project Structure

```
ai_sport_news_chatbot/
├── src/
│   ├── chat.py                          # Main chat interface (CLI)
│   ├── rag/
│   │   ├── rag_update_news.py           # RAG update logic
│   │   ├── crawler/
│   │   │   └── news_crawler.py          # News crawler v1
│   │   └── news_pipeline/
│   │       ├── cleaner.py               # Data cleaning
│   │       ├── chunker.py               # Text chunking
│   │       └── vector_store.py          # Vector database operations
│   └── util/
│       └── tools.py                     # Utility tools
├── tests/
│   └── test_hello.py                    # Test file
├── logs/                                # Application logs
├── .vscode/
│   └── settings.json                    # VSCode settings
├── .dockerignore                        # Docker ignore file
├── .gitignore                           # Git ignore file
├── docker-compose.yml                   # Docker compose configuration
├── Dockerfile                           # Docker configuration
├── requirements.txt                     # Python dependencies
├── plan.md                              # Development plan and roadmap
└── README.md                            # This file
```

---

## Troubleshooting

### Database Connection Error
- Ensure Docker Desktop is running
- Check if port 5432 is available
- Verify `.env` database URL is correct

### API Key Errors
- Confirm API keys are valid and active
- Check for typos in `.env` file

### Module Not Found Errors
- Run `pip install -r requirements.txt`
- Ensure you're using the correct Python virtual environment

---

## Next Steps (short-term priorities)

- [ ] Finish Phase 1 bug fixes (Wikipedia fallback, prompt isolation, OLD doc warnings)
- [ ] Add conversational memory injection to the generation node (inject last 3–4 turns)
- [ ] Add automated ingestion/scheduling for the RAG update flow
- [ ] Add unit tests and an evaluation harness for hallucination measurement

---

## Contributing

Please open issues or PRs for bugs and enhancements. If you want to help with Phase 1 stabilization or Phase 2 evaluation tooling, mention it in an issue and assign yourself.
