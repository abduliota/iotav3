# IOTA v3 - SAMA/NORA Regulatory Chatbot

FastAPI + Next.js RAG chatbot for Saudi regulatory and compliance Q&A (SAMA/NORA context), with streaming responses, source citations, and session persistence.

## Features

- Retrieval-Augmented Generation (RAG) over chunked regulatory documents
- English + Arabic support through multilingual embeddings
- Streaming chat responses over NDJSON (`/api/query-stream`)
- Source metadata in answers (`document_name`, pages, similarity, snippet)
- Session persistence and message history via Supabase tables
- Feedback capture (like/dislike) for responses
- Redis cache support with in-memory fallback
- Next.js frontend with fingerprint-based anonymous identity/session continuity

## Tech Stack

- **Backend**: FastAPI, Uvicorn, Pydantic
- **Frontend**: Next.js 15, React 19, TypeScript, Tailwind CSS
- **Vector/Data**: Supabase (Postgres + pgvector + RPCs)
- **Embeddings**: `intfloat/multilingual-e5-small` (384-dim)
- **LLM Backends**: `qwen` (local), `openai`, `azure`
- **Caching**: Redis (optional), in-memory fallback

## Repository Structure

```text
iotav3/
├── backend/
│   ├── api.py                 # FastAPI app + all API routes
│   ├── simple_rag.py          # Core RAG pipeline
│   ├── diagnostic.py          # Health checks for DB/embeddings/retrieval
│   ├── test_questions.py      # Integration-style Q&A tests
│   ├── requirements.txt       # Python dependencies
│   ├── Dockerfile             # Backend container image
│   └── .env                   # Backend env vars (local only)
├── frontend/
│   ├── src/app/page.tsx       # Main chat UI
│   ├── src/hooks/useChat.ts   # Chat state + streaming logic
│   ├── src/lib/api.ts         # Frontend API client wrappers
│   ├── src/app/api/chat/route.ts # Optional proxy to backend stream
│   ├── package.json
│   └── .env.local             # Frontend env vars (local only)
└── README.md
```

## How It Works

1. User sends a query from the Next.js app.
2. Frontend streams from backend `POST /api/query-stream`.
3. Backend runs retrieval/generation in `simple_rag.py`.
4. Backend streams NDJSON events:
   - `token` (incremental text)
   - `sources` (final metadata + message id)
   - `done`
5. Interaction can be persisted to Supabase session tables.

## Prerequisites

- Python 3.10+ (3.11 recommended)
- Node.js 18+ (Node 20 recommended)
- npm (repo already has `package-lock.json`)
- Supabase project with required schema/RPCs
- (Optional) Redis instance for persistent semantic cache

## Local Development

### 1) Clone

```bash
git clone https://github.com/abduliota/iotav3.git
cd iotav3
```

### 2) Backend Setup

```bash
cd backend
python -m venv .venv
```

Activate virtualenv:

- Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

- Windows (Git Bash):

```bash
source .venv/Scripts/activate
```

- macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create backend env file:

```bash
# In backend/
cp .env.example .env
```

If `.env.example` does not exist, create `.env` manually (see Environment Variables section).

Run diagnostics:

```bash
python diagnostic.py
```

Start API:

```bash
python api.py
```

Backend will run on `http://localhost:8000`.

### 3) Frontend Setup

Open a second terminal:

```bash
cd frontend
npm install
```

Set frontend env (`frontend/.env.local`):

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Start frontend:

```bash
npm run dev
```

Frontend will run on `http://localhost:3000`.

## Environment Variables

### Backend (`backend/.env`)

Required (typical):

| Variable | Purpose |
|---|---|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` (or `SUPABASE_KEY`) | Service key for backend DB access |
| `LLM_BACKEND` | `qwen`, `openai`, or `azure` |
| `EMBEDDING_MODEL` | Embedding model name |
| `EMBEDDING_DIM` | Embedding dimension (expected 384) |
| `CORS_ORIGINS` | Allowed frontend origins (comma-separated) |

Common optional:

| Variable | Purpose |
|---|---|
| `TOP_K` | Retrieval chunk count |
| `SIMILARITY_THRESHOLD` | Retrieval similarity floor |
| `LOW_CONF_THRESHOLD` | Confidence guard threshold |
| `CACHE_BACKEND` | `memory` or `redis` |
| `REDIS_URL` | Redis connection string |
| `CACHE_SIMILARITY_THRESH` | Similarity threshold for semantic cache hit |
| `ADMIN_API_KEY` | Protects cache-clear endpoint |
| `OPENAI_API_KEY` | Needed when `LLM_BACKEND=openai` |
| Azure OpenAI vars | Needed when `LLM_BACKEND=azure` |

### Frontend (`frontend/.env.local`)

| Variable | Purpose |
|---|---|
| `NEXT_PUBLIC_API_URL` | Base URL for FastAPI backend |

## API Reference

Base URL (local): `http://localhost:8000`

### Health

- `GET /health`

### Query (JSON)

- `POST /api/query`
- Body:

```json
{
  "query": "What are AML requirements?",
  "user_id": "optional-user-id",
  "session_id": "optional-session-id",
  "top_k": 5,
  "debug": false
}
```

### Query (Streaming)

- `POST /api/query-stream`
- Returns `application/x-ndjson`
- Event shapes:

```json
{"type":"token","text":"..."}
{"type":"sources","sources":[...],"message_id":"...","cached":false,"method":"generative"}
{"type":"done"}
```

Error event:

```json
{"type":"error","message":"..."}
```

### Feedback

- `POST /api/feedback`
- Body:

```json
{
  "session_id": "session-id",
  "user_id": "user-id",
  "message_id": "message-id",
  "feedback": 1,
  "comments": "optional",
  "user_message": "optional",
  "assistant_message": "optional"
}
```

### Session Messages

- `GET /api/session/{session_id}/messages?limit=20`

### Cache Admin

- `GET /admin/cache/status`
- `POST /admin/cache/clear?api_key=...`

## Testing

Backend integration tests (API should be running on port 8000):

```bash
cd backend
python test_questions.py
```

## Deployment

### Render (Web Service)

Use a **Web Service** (not Static Site / Worker) for backend API.

- **Root Directory**: `backend`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`
- **Health Check Path**: `/health`

Set required backend environment variables in Render dashboard.

### Docker (Backend)

From repository root:

```bash
docker build -f backend/Dockerfile -t sama-chatbot .
docker run -p 8000:8000 --env-file backend/.env sama-chatbot
```

## Troubleshooting

- **CORS errors**: set `CORS_ORIGINS` to include your frontend URL(s).
- **No retrieval hits / weak answers**: ensure embedding model and dimensions match indexed vectors.
- **Streaming issues**: verify client consumes NDJSON line-by-line and backend responds with `application/x-ndjson`.
- **Slow or heavy local model**: if using `qwen`, consider switching to `openai`/`azure` backend for hosted inference.
- **Session history empty**: verify Supabase session tables exist and service key has insert/select access.

## Notes

- Keep secrets in `.env`/`.env.local`; do not commit credentials.
- The repository `.gitignore` is configured to ignore common local artifacts/logs.
