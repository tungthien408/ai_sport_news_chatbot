### **Phase 1: Stabilization & Conversational Memory**

*Focus: Fix existing blind spots and make the bot context-aware.*

* **Implement Bug Fixes:** Apply the four critical fixes Claude identified. Specifically, update the Wikipedia fallback to use `search()` before `summary()`, isolate the user prompt instructions in English to prevent the model from code-switching, and enforce the explicit warning for `OLD` documents.
* **Conversational Memory (The Missing Link):** Currently, `chat.py` accumulates `message_history`, but `generate_answer` in `graph_nodes.py` creates a fresh `[SystemMessage, HumanMessage]` pair every time. Update the generation node to inject the last 3-4 conversational turns into the context so the LLM can handle follow-up questions gracefully (e.g., "What was the score?" -> "Who scored the winning goal?").

### **Phase 2: Agentic Reflection & Quantitative Evaluation**

*Focus: Apply DeepLearning.AI concepts and mathematically prove they work.*

* **The Reflection Pattern:** Add a new LangGraph node immediately after `generate_answer`. Have a second LLM prompt act as a "Critic" to ask: *"Does this answer contain any information not explicitly found in the retrieved documents?"* If yes, loop back and rewrite. This is a genuinely agentic feature that directly addresses hallucinations.
* **LLM-as-a-Judge Evals:** Write a JSON file with 20-30 test questions (some historical, some real-time, some off-topic) and their expected answers. Write a script to measure the chatbot's accuracy and hallucination rate *before* and *after* implementing the Reflection node. Committing these metrics to your repository proves you understand AI system evaluation.

### **Phase 3: Production Backend & Persistence**

*Focus: Move from a CLI script to a scalable service architecture.*

* **FastAPI Integration:** Wrap your compiled LangGraph workflow in robust FastAPI endpoints (e.g., `POST /chat`, `GET /history/{session_id}`). Use Pydantic models for strict input validation.
* **Session Persistence:** Right now, memory is lost when the terminal closes. Use PostgreSQL and SQLAlchemy to persist user sessions and conversation histories.
* **Dockerization:** Update your `docker-compose.yml` so that a single command spins up the FastAPI backend, the PostgreSQL memory database, and the PGVector retrieval database simultaneously.

### **Phase 4: Full-Stack Presentation & Domain Expansion**

*Focus: Finalize the portfolio piece with a polished user experience and diverse data.*

* **Custom UI:** To showcase full-stack capabilities, build a sleek, interactive chat interface using React.js to consume your FastAPI endpoints. This stands out significantly more in a repository than a basic Streamlit dashboard.
* **Data Diversification:** Prove that your RAG pipeline and crawler are scalable by expanding the sources. Integrating RSS feeds for other active interests—such as global badminton tournament updates alongside standard football news—demonstrates that your architecture can seamlessly handle and categorize multiple distinct sports domains at once.

