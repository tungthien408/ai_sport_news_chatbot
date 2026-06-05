## **Plan: LangGraph Workflow for Sport News Chatbot**

### **TL;DR**
Bạn sẽ convert workflow từ **prompt-based agent** thành **explicit graph** với 8 nodes rõ ràng + conditional routing. Agent sẽ vẫn gọi tool nhưng trong context cụ thể của từng node. Điều này giúp **debug dễ hơn**, **kiểm soát flow rõ ràng**, và **đảm bảo logic đúng theo system prompt**.

---

### **Graph Structure**

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  START (user question)                                          │
│    ↓                                                             │
│  [1] process_input (khởi tạo state)                             │
│    ↓                                                             │
│  [2] retrieve_initial (gọi retrieve_context)                    │
│    ├─→ docs found? → [3] check_recency                          │
│    └─→ no docs    → [4] decide_crawl                            │
│                                                                 │
│  [3] check_recency (kiểm tra doc ngày)                          │
│    ├─→ recent       → [7] generate_answer                       │
│    └─→ old + exist  → ask user (A or B)                         │
│           ├─→ A (crawl)  → [5] crawl_and_process                │
│           └─→ B (skip)   → [7] generate_answer                  │
│                                                                 │
│  [4] decide_crawl (lựa chọn source: user URL/VnExpress/BBC)    │
│    ├─→ crawl_count < 2 → [5] crawl_and_process                  │
│    └─→ crawl_count ≥ 2 → [7] generate_answer (no more crawls)   │
│                                                                 │
│  [5] crawl_and_process                                          │
│    ├─→ success → [6] retrieve_after_crawl                       │
│    └─→ fail    → [7] generate_answer (fallback)                 │
│                                                                 │
│  [6] retrieve_after_crawl (lấy docs mới sau crawl)              │
│    ├─→ new docs recent → [7] generate_answer                   │
│    └─→ no new docs or old → [3] check_recency (loop) OR         │
│        decide to crawl next source (if < 2)                     │
│                                                                 │
│  [7] generate_answer (tạo response + citations)                 │
│    ↓                                                             │
│  END (return to user)                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### **Steps**

#### **Phase 1: Setup & Initialization** *(Parallelize file reads with data structure design)*
1. Create `src/graph_state.py` — Define `GraphState` TypedDict with fields:
   - `messages`: message history
   - `current_question`: user's question
   - `time_preference_hours`: parsed from user input (default 12)
   - `docs_found`: list of retrieved documents
   - `recency_status`: "RECENT" | "OLD" | "NOT_FOUND"
   - `crawl_history`: list of attempted sources (e.g., `["vnexpress.net"]`)
   - `crawl_count`: int (stop at 2)
   - `user_feedback`: "A" | "B" | None

#### **Phase 2: Core Nodes**  *(Implement node functions in parallel groups)*

**Group 2A — Data retrieval nodes** (can work in parallel once state is ready):

2. Node `process_input` — Parse user question, extract time preference ("mới nhất"→12h, "24h"→24, etc). Initialize state.

3. Node `retrieve_initial` — Call `retrieve_context(question)`, populate `docs_found`.

4. Node `check_recency` — Compare `docs_found[].published_at` with cutoff time. Set `recency_status`.

**Group 2B — Crawl decision & execution**:

5. Node `decide_crawl` — Logic: user provided URL? → use that; else if VnExpress not crawled → use VnExpress; else BBC. Check `crawl_count < 2`.

6. Node `crawl_and_process` — Call `crawl_news_feed(url)`, track in `crawl_history`, increment `crawl_count`.

7. Node `retrieve_after_crawl` — Re-call `retrieve_context(question)`, compare new docs.

**Group 2C — Response generation**:

8. Node `generate_answer` — LLM synthesizes response using:
   - `docs_found` (if recent) or asks user if old
   - Add citations (title + URL + date)
   - Clear language ("thông tin lịch sử từ ngày..." if using old data)

---

#### **Phase 3: Conditional Edges** *(Define routing logic)*
9. Define `should_check_recency(state)` → `"check_recency"` if docs found else `"decide_crawl"`

10. Define `should_crawl_after_recency(state)` → `"crawl_and_process"` if old + need crawl; else `"generate_answer"`

11. Define `should_crawl_after_retrieve(state)` → loop back to `check_recency` if new docs, or stop

12. Define `should_continue_crawling(state)` → allow crawl if `crawl_count < 2` else stop

---

#### **Phase 4: User Interaction** *(Optional interactive branch)*
13. If `recency_status == "OLD"` && docs exist (not empty), send "Ask user: (A) crawl / (B) use old?"
    - Capture user response → set `user_feedback`
    - Route accordingly

---

#### **Phase 5: Update chat.py** *(Integrate graph into main loop)*
14. Replace `agent.invoke()` with `graph.invoke()` or `graph.stream()`

15. Pass `{"messages": [...], "current_question": question}` as input

16. Retrieve final answer from `state["messages"]` (last AI message)

---

### **Relevant Files**
- chat.py — Current agent setup (will refactor)
- tools.py — Tools (reuse as-is: `retrieve_context`, `crawl_news_feed`, `current_time`)
- rag_update_news.py — Pipeline logic (used by `crawl_news_feed`)

**Files to create:**
- `src/graph_state.py` — State schema
- `src/graph_nodes.py` — Node implementations
- `src/graph_edges.py` — Edge routing logic
- `src/graph_builder.py` — Assemble graph (StateGraph + add_node + add_edge + compile)

---

### **Verification**

1. **Unit test each node** — Mock state, verify output (e.g., `process_input` correctly parses "12h")

2. **Test edge conditions** — Verify conditional routing (recent docs → skip crawl; old docs → ask user)

3. **End-to-end test** — Run full flow with sample questions:
   - "Tin mới nhất Real Madrid" (recent docs available → should use + answer)
   - "Tin Real Madrid trong 1 ngày qua" (old docs → ask user or crawl)
   - "Tin về Jose Mourinho" (no docs → crawl + retrieve + answer)

4. **Log each node transition** — Print state at each node to verify flow

---

### **Decisions & Scope**

✅ **Included:**
- Complex decision tree matching system prompt logic
- State tracking for crawl history + time preference
- Agent within nodes (not explicit tool calls, but structured prompt per node)
- Citation formatting in response
- Loop-back for recency recheck

❌ **Out of scope (next phase):**
- Parallel tool execution (e.g., crawl VnExpress + BBC simultaneously)
- Caching results between queries
- User session persistence across multiple questions

---

**Tiếp theo?** Bạn đã ready để bắt đầu build graph từ node `process_input` không, hay bạn muốn tôi refine thêm những điểm nào? 🎯