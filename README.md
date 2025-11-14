# SagePick.ai Services

This repository hosts two FastAPI-based microservices:

- **Train service** under `train/` handles model training, evaluation, and artifact publishing.
- **Inference service** under `inference/` serves recommendations and orchestrates agentic research flows behind an Nginx reverse proxy.

Both services now ship with production-ready Dockerfiles and `docker-compose` stacks so you can build and run them with a single command.

## Prerequisites

- Docker Engine 24+
- Docker Compose V2 (`docker compose` CLI)
- Copy each service's environment template before running:
  - `train/.env.example → train/.env`
  - `inference/.env.example → inference/.env`

## Train Service

```bash
cd train
cp .env.example .env  # adjust credentials/models
docker compose up -d --build
```

- The API becomes available at <http://localhost:8000>.
- Source code lives in `/app` inside the container, and local `train/data/` is mounted to `/app/data` for ad-hoc datasets.
- Logs: `docker compose logs -f train-service` (from `train/`).
- Stop & remove: `docker compose down`.

## Inference Service (with Nginx)

```bash
cd inference
cp .env.example .env  # or keep your existing .env
docker compose up -d --build
```

- `inference-app` runs Uvicorn on port 8001 inside the network.
- `inference-proxy` exposes port 80 publicly and applies the hardened Nginx config in `inference/nginx/nginx.conf`, which:
  - Forces HTTP on port 80 only.
  - Extends agent-friendly timeouts (600s connect/read/send) for long-running graph calls.
  - Expands proxy buffers/cache (2 GB cap) for large payloads and streaming writes.
- Reach the API through <http://localhost> (docs at `/docs`).
- Tail proxies: `docker compose logs -f inference-proxy` (from `inference/`).

## Helpful Commands

| Action | Train | Inference |
| --- | --- | --- |
| Rebuild after code changes | `docker compose up -d --build` | `docker compose up -d --build` |
| View health endpoint | `curl http://localhost:8000/health` | `curl http://localhost/health` |
| Stop services | `docker compose down` | `docker compose down` |

## Next Steps

- Wire CI to run `docker compose config` for both stacks to catch regressions.
- Optionally push the images to a registry by adding `image:` tags inside each compose file.
