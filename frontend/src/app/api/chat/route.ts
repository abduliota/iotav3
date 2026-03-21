/**
 * app/api/chat/route.ts
 *
 * Next.js server-side proxy for the FastAPI streaming endpoint.
 * The browser calls /api/chat (same origin) — no CORS issues.
 * This route forwards to FastAPI and streams the response back.
 */
import { NextRequest } from 'next/server'

const BACKEND = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export async function POST(req: NextRequest) {
  const body = await req.json()

  const upstream = await fetch(`${BACKEND}/api/query-stream`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
  })

  if (!upstream.ok) {
    return new Response(
      JSON.stringify({ error: `Backend error: ${upstream.status}` }),
      { status: upstream.status, headers: { 'Content-Type': 'application/json' } }
    )
  }

  // Stream the NDJSON response straight through to the browser
  return new Response(upstream.body, {
    status:  200,
    headers: {
      'Content-Type':     'application/x-ndjson',
      'X-Accel-Buffering': 'no',
      'Cache-Control':    'no-cache',
    },
  })
}
