# Rahul12 AI Control Center v2

A mobile-first AI control plane with an original animated network hero, chat commands, live workflow routing, Kaggle worker telemetry, logs, research APIs, dataset jobs and a small provider fallback chain.

## Architecture

Browser -> FastAPI control plane -> AI/research services
                              ^
                              |
                         Kaggle worker

The browser never receives provider keys or the Kaggle token. The Kaggle notebook/worker uses `WORKER_TOKEN` to authenticate to the control plane.

## What is included

- `web/index.html` — mobile-first Meta-inspired visual language with an original Canvas network animation. It is not copied media.
- `backend/main.py` — FastAPI control plane, chat router, AI fallback, research, task queue, worker telemetry, logs and WebSocket updates.
- `kaggle_worker/worker.py` — Kaggle-side worker with CPU/RAM/GPU telemetry and staged workflow reporting.
- `backend/.env.example` — server-side configuration names only.

## Provider routing

Small fallback chain:
1. NVIDIA #1
2. NVIDIA #2
3. GLM
4. OpenRouter
5. Groq
6. DeepSeek

Model IDs are configurable. Do not assume a model is available in your account.

DeepSeek base URL is configured as `https://api.b.ai/v1`; the worker/backend appends `/chat/completions`.

## Research

Tavily provides web research and YouTube Data API v3 provides video search. The model receives those results and is instructed not to invent source links.

## Secrets

Never put API keys in `index.html`, GitHub, or a public dataset. Put server secrets in your hosting provider's secret store. Put `WORKER_TOKEN` in Kaggle Secrets and the same value in the backend's secret store.

If a real token was ever pasted into a chat, GitHub issue, screenshot, or public repo, revoke/rotate it before production use.

## Kaggle setup

1. Create a Kaggle Notebook.
2. Add Secrets: `CONTROL_URL`, `WORKER_TOKEN`.
3. Install requirements from `kaggle_worker/requirements.txt`.
4. Run `python /kaggle/working/worker.py` (or paste the worker into a notebook cell and run it).
5. Keep the notebook session running while you want live control. The web app can only control a running worker; it cannot make Kaggle provide unlimited compute or bypass Kaggle quotas.

## Backend setup

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --host 0.0.0.0 --port 8000
```

Serve `web/index.html` from the same origin in production.

## GitHub

The optional `GITHUB_TOKEN` is server-side only. Keep the repository integration read-only until you explicitly implement and secure write operations.

## MonkeyCode

MonkeyCode is a separate open-source coding platform. Its repository is AGPL-3.0. If you integrate or modify its source, keep the license and source-disclosure requirements in mind. Prefer a clean adapter boundary rather than copying large parts of the project into this control plane.

## Production hardening

For a public deployment add real authentication, HTTPS, rate limiting, a persistent queue (Redis/Postgres), per-user authorization, audit logs, a managed secret store, signed worker registration and strict task allowlists.
