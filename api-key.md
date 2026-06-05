# SAMA NORA — API Key Management

**Base URL:** `https://iotaregtech-gbffd4bzgwcsa6e2.southindia-01.azurewebsites.net`

---

## How it works

Every request to `/api/*` endpoints must include the `X-API-Key` header.  
Each client/partner gets their own unique key. Keys are stored in Supabase (`api_keys` table).

```
Client → POST /api/query + X-API-Key: their-key → ✅ 200 answer
Client → POST /api/query (no key)                → ❌ 403 Forbidden
```

---

## Admin key setup

All admin endpoints require `?admin_key=YOUR_ADMIN_KEY` once `ADMIN_API_KEY` is set in Azure.

**To set it — run this locally:**
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```
Then add to Azure → App Service → Configuration → New setting:
- **Name:** `ADMIN_API_KEY`
- **Value:** the output above

---

## List all keys

```bash
curl "https://iotaregtech-gbffd4bzgwcsa6e2.southindia-01.azurewebsites.net/admin/keys?admin_key=YOUR_ADMIN_KEY"
```

**Response:**
```json
{
  "keys": [
    {
      "id": "72936def-a57e-4fef-b69e-e805355362fd",
      "label": "ZetaLabs Internal",
      "is_active": true,
      "created_at": "2026-06-05T09:32:18.410899+00:00"
    }
  ]
}
```

> Raw key values are **never returned** after creation — only metadata.

---

## Create a new key

```bash
curl -X POST "https://iotaregtech-gbffd4bzgwcsa6e2.southindia-01.azurewebsites.net/admin/keys?admin_key=YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"label\": \"Client A\"}"
```

**Response:**
```json
{
  "api_key": "abc123...64chars...",
  "id": "uuid-here",
  "label": "Client A",
  "created_at": "2026-06-05T...",
  "warning": "Store this key securely. It will not be shown again."
}
```

> ⚠️ Copy the `api_key` immediately — it is shown **once only** and never stored in plain text.

---

## Revoke a key

**Step 1 — List keys to get the UUID:**
```bash
curl "https://iotaregtech-gbffd4bzgwcsa6e2.southindia-01.azurewebsites.net/admin/keys?admin_key=YOUR_ADMIN_KEY"
```

**Step 2 — Revoke by UUID:**
```bash
curl -X DELETE "https://iotaregtech-gbffd4bzgwcsa6e2.southindia-01.azurewebsites.net/admin/keys/UUID-HERE?admin_key=YOUR_ADMIN_KEY"
```

**Response:**
```json
{ "status": "revoked", "id": "UUID-HERE" }
```

> The key is deactivated instantly. The next request using that key returns `403`. The record is kept in the DB for audit purposes.

---

## Using a key (client side)

Add the `X-API-Key` header to every request:

```bash
curl -X POST "https://iotaregtech-gbffd4bzgwcsa6e2.southindia-01.azurewebsites.net/api/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d "{\"query\": \"What are the AML requirements?\", \"top_k\": 5}"
```

---

## Current keys

| Label | ID | Status |
|---|---|---|
| ZetaLabs Internal | `72936def-a57e-4fef-b69e-e805355362fd` | ✅ Active |

> ZetaLabs Internal key is used by the Vercel frontend via `VITE_API_KEY` env var.

---

## Quick reference

| Action | Command |
|---|---|# SAMA NORA — API Key Management

**Base URL:** `https://iotaregtech-gbffd4bzgwcsa6e2.southindia-01.azurewebsites.net`

---

## How it works

Every request to `/api/*` endpoints must include the `X-API-Key` header.  
Each client/partner gets their own unique key. Keys are stored in Supabase (`api_keys` table).

```
Client → POST /api/query + X-API-Key: their-key → ✅ 200 answer
Client → POST /api/query (no key)                → ❌ 403 Forbidden
```

---

## Admin key setup

All admin endpoints require `?admin_key=YOUR_ADMIN_KEY` once `ADMIN_API_KEY` is set in Azure.

**To set it — run this locally:**
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```
Then add to Azure → App Service → Configuration → New setting:
- **Name:** `ADMIN_API_KEY`
- **Value:** the output above

---

## List all keys

```bash
curl "https://iotaregtech-gbffd4bzgwcsa6e2.southindia-01.azurewebsites.net/admin/keys?admin_key=YOUR_ADMIN_KEY"
```

**Response:**
```json
{
  "keys": [
    {
      "id": "72936def-a57e-4fef-b69e-e805355362fd",
      "label": "ZetaLabs Internal",
      "is_active": true,
      "created_at": "2026-06-05T09:32:18.410899+00:00"
    }
  ]
}
```

> Raw key values are **never returned** after creation — only metadata.

---

## Create a new key

```bash
curl -X POST "https://iotaregtech-gbffd4bzgwcsa6e2.southindia-01.azurewebsites.net/admin/keys?admin_key=YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"label\": \"Client A\"}"
```

**Response:**
```json
{
  "api_key": "abc123...64chars...",
  "id": "uuid-here",
  "label": "Client A",
  "created_at": "2026-06-05T...",
  "warning": "Store this key securely. It will not be shown again."
}
```

> ⚠️ Copy the `api_key` immediately — it is shown **once only** and never stored in plain text.

---

## Revoke a key

**Step 1 — List keys to get the UUID:**
```bash
curl "https://iotaregtech-gbffd4bzgwcsa6e2.southindia-01.azurewebsites.net/admin/keys?admin_key=YOUR_ADMIN_KEY"
```

**Step 2 — Revoke by UUID:**
```bash
curl -X DELETE "https://iotaregtech-gbffd4bzgwcsa6e2.southindia-01.azurewebsites.net/admin/keys/UUID-HERE?admin_key=YOUR_ADMIN_KEY"
```

**Response:**
```json
{ "status": "revoked", "id": "UUID-HERE" }
```

> The key is deactivated instantly. The next request using that key returns `403`. The record is kept in the DB for audit purposes.

---

## Using a key (client side)

Add the `X-API-Key` header to every request:

```bash
curl -X POST "https://iotaregtech-gbffd4bzgwcsa6e2.southindia-01.azurewebsites.net/api/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d "{\"query\": \"What are the AML requirements?\", \"top_k\": 5}"
```

---

## Current keys

| Label | ID | Status |
|---|---|---|
| ZetaLabs Internal | `72936def-a57e-4fef-b69e-e805355362fd` | ✅ Active |

> ZetaLabs Internal key is used by the Vercel frontend via `VITE_API_KEY` env var.

---

## Quick reference

| Action | Command |
|---|---|
| List keys | `GET /admin/keys?admin_key=...` |
| Create key | `POST /admin/keys?admin_key=...` + `{"label": "..."}` |
| Revoke key | `DELETE /admin/keys/{id}?admin_key=...` |
| Use key | Add `X-API-Key: ...` header to any `/api/*` request |

---

## Supabase table

```sql
CREATE TABLE api_keys (
  id         uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  api_key    text UNIQUE NOT NULL,
  label      text NOT NULL,
  is_active  boolean DEFAULT true,
  created_at timestamptz DEFAULT now()
);
```

Keys can also be managed directly in the Supabase dashboard under the `api_keys` table.# SAMA NORA — API Key Management

**Base URL:** `https://iotaregtech-gbffd4bzgwcsa6e2.southindia-01.azurewebsites.net`

---

## How it works

Every request to `/api/*` endpoints must include the `X-API-Key` header.  
Each client/partner gets their own unique key. Keys are stored in Supabase (`api_keys` table).

```
Client → POST /api/query + X-API-Key: their-key → ✅ 200 answer
Client → POST /api/query (no key)                → ❌ 403 Forbidden
```

---

## Admin key setup

All admin endpoints require `?admin_key=YOUR_ADMIN_KEY` once `ADMIN_API_KEY` is set in Azure.

**To set it — run this locally:**
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```
Then add to Azure → App Service → Configuration → New setting:
- **Name:** `ADMIN_API_KEY`
- **Value:** the output above

---

## List all keys

```bash
curl "https://iotaregtech-gbffd4bzgwcsa6e2.southindia-01.azurewebsites.net/admin/keys?admin_key=YOUR_ADMIN_KEY"
```

**Response:**
```json
{
  "keys": [
    {
      "id": "72936def-a57e-4fef-b69e-e805355362fd",
      "label": "ZetaLabs Internal",
      "is_active": true,
      "created_at": "2026-06-05T09:32:18.410899+00:00"
    }
  ]
}
```

> Raw key values are **never returned** after creation — only metadata.

---

## Create a new key

```bash
curl -X POST "https://iotaregtech-gbffd4bzgwcsa6e2.southindia-01.azurewebsites.net/admin/keys?admin_key=YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"label\": \"Client A\"}"
```

**Response:**
```json
{
  "api_key": "abc123...64chars...",
  "id": "uuid-here",
  "label": "Client A",
  "created_at": "2026-06-05T...",
  "warning": "Store this key securely. It will not be shown again."
}
```

> ⚠️ Copy the `api_key` immediately — it is shown **once only** and never stored in plain text.

---

## Revoke a key

**Step 1 — List keys to get the UUID:**
```bash
curl "https://iotaregtech-gbffd4bzgwcsa6e2.southindia-01.azurewebsites.net/admin/keys?admin_key=YOUR_ADMIN_KEY"
```

**Step 2 — Revoke by UUID:**
```bash
curl -X DELETE "https://iotaregtech-gbffd4bzgwcsa6e2.southindia-01.azurewebsites.net/admin/keys/UUID-HERE?admin_key=YOUR_ADMIN_KEY"
```

**Response:**
```json
{ "status": "revoked", "id": "UUID-HERE" }
```

> The key is deactivated instantly. The next request using that key returns `403`. The record is kept in the DB for audit purposes.

---

## Using a key (client side)

Add the `X-API-Key` header to every request:

```bash
curl -X POST "https://iotaregtech-gbffd4bzgwcsa6e2.southindia-01.azurewebsites.net/api/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d "{\"query\": \"What are the AML requirements?\", \"top_k\": 5}"
```

---

## Current keys

| Label | ID | Status |
|---|---|---|
| ZetaLabs Internal | `72936def-a57e-4fef-b69e-e805355362fd` | ✅ Active |

> ZetaLabs Internal key is used by the Vercel frontend via `VITE_API_KEY` env var.

---

## Quick reference

| Action | Command |
|---|---|
| List keys | `GET /admin/keys?admin_key=...` |
| Create key | `POST /admin/keys?admin_key=...` + `{"label": "..."}` |
| Revoke key | `DELETE /admin/keys/{id}?admin_key=...` |
| Use key | Add `X-API-Key: ...` header to any `/api/*` request |

---

## Supabase table

```sql
CREATE TABLE api_keys (
  id         uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  api_key    text UNIQUE NOT NULL,
  label      text NOT NULL,
  is_active  boolean DEFAULT true,
  created_at timestamptz DEFAULT now()
);
```

Keys can also be managed directly in the Supabase dashboard under the `api_keys` table.
| List keys | `GET /admin/keys?admin_key=...` |
| Create key | `POST /admin/keys?admin_key=...` + `{"label": "..."}` |
| Revoke key | `DELETE /admin/keys/{id}?admin_key=...` |
| Use key | Add `X-API-Key: ...` header to any `/api/*` request |

---

## Supabase table

```sql
CREATE TABLE api_keys (
  id         uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  api_key    text UNIQUE NOT NULL,
  label      text NOT NULL,
  is_active  boolean DEFAULT true,
  created_at timestamptz DEFAULT now()
);
```

Keys can also be managed directly in the Supabase dashboard under the `api_keys` table.