# AI Sport News Chatbot

An LLM-powered chatbot that answers user queries about sports news by crawling, processing, and retrieving information from a vector database.

**Current Status:** Data pipeline in development (crawling → cleaning → chunking → vector DB insertion)

<img width="1031" height="281" alt="image" src="https://github.com/user-attachments/assets/91fcb34c-237a-49df-850a-a70427df724a" />


---

## Prerequisites

- **Python** 3.8 or higher
- **Docker Desktop** (running)
- **API Keys:** QWEN and NVIDIA (see [API Setup](#api-setup))

---

## Quick Start

### 1. Prepare Your Environment

```bash
cd ./ai_sport_news_chatbot
mkdir ./logs
cp .env.example .env  # or create manually (see step 3)
```

### 2. Start Docker Services

Ensure Docker Desktop is running, then:

```bash
docker compose up -d db
# Wait for the database to be ready (~10-15 seconds)
```

### 3. Configure Environment Variables

Edit the `.env` file and add your API keys:

```env
DB_NAME="vectordb"
DB_URL="postgresql+psycopg://langchain:langchain@localhost:5432/vectordb"

QWEN_API_KEY=your_actual_key_here
NVIDIA_API_KEY=your_actual_key_here
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
python ./src/chat.py
```

---

## API Setup

### Getting API Keys

- **QWEN_API_KEY:** [Get from Alibaba Cloud Qwen](https://dashscope.console.aliyun.com/)
- **NVIDIA_API_KEY:** [Get from NVIDIA API Catalog](https://build.nvidia.com/)

---

## Project Structure

```
ai_sport_news_chatbot/
├── src/
│   ├── chat.py                          # Main chat interface
│   ├── rag/
│   │   ├── rag_update_news.py           # RAG update logic
│   │   ├── crawler/
│   │   │   ├── news_crawler.py          # News crawler v1
│   │   │   └── news_crawler_v2.py       # News crawler v2
│   │   └── news_pipeline/
│   │       ├── cleaner.py               # Data cleaning
│   │       ├── chunker.py               # Text chunking
│   │       └── vector_store.py          # Vector database operations
│   └── util/
│       ├── tools.py                     # Utility tools
│       └── tool_old.py                  # Legacy tools
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

## Next Steps

- [ ] Complete data crawling pipeline
- [ ] Implement data cleaning and chunking
- [ ] Test vector database queries
- [ ] Optimize chatbot response quality

---

## Contributing

Feel free to submit issues and enhancement requests!
