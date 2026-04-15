# IOTA Frontend2

`frontend2` is a Vite + React TypeScript frontend aligned to the existing backend API contract in `backend/api.py`.

## Backend Contract Used

- `GET /health`
- `POST /api/query`
- `GET /api/documents`
- `GET /api/conversations`
- `GET /api/session/{session_id}/messages`
- `POST /api/feedback`

Note: Upload and OTP authentication endpoints are intentionally not used because they are not available in the current backend.

## Environment

Create `.env` from `.env.example`:

```bash
cp .env.example .env
```

Use the following values:

```env
VITE_API_BASE=/backend
VITE_UPLOAD_ENABLED=false
VITE_REQUIRE_UPLOAD_AUTH=false
```

`/backend` uses Vite dev proxy to forward requests to `http://localhost:8000`, which avoids browser CORS/preflight issues during development.

## Run

```bash
npm install
npm run dev
```

## CORS Note

When using `VITE_API_BASE=/backend` in dev, CORS is typically bypassed through the proxy.
If you switch `VITE_API_BASE` to a direct backend origin, ensure backend `CORS_ORIGINS` includes the frontend host (`http://localhost:5173`).
