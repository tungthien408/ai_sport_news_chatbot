# AI Sport News Chatbot

**Current Scope:** Finishing data pipeline (Crawl newspaper -> Cleaning -> Chunking -> Inserting into vector database)

**Goal:** Implementing an LLM chatbot that answers user's sport-related queries.

---

## Steps to Run the Project

### Step 1: Open Docker Desktop

Make sure Docker Desktop is running.

---

### Step 2: Run the Following Commands

```cmd
cd ./ai_sport_news_chatbot
code .env
mkdir ./logs
docker compose up -d db
```

---

### Step 3: Configure the `.env` File

Ensure the `.env` file contains the following:

```.env
DB_NAME="vectordb"
DB_URL="postgresql+psycopg://langchain:langchain@localhost:5432/vectordb"

QWEN_API_KEY=value
NVIDIA_API_KEY=value
```

Replace `value` with your actual API keys.

---

### Step 4: Run the Application

Execute the following command:

```cmd
python ./src/main.py
```