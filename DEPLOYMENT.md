# Deployment guide

The dashboard ships as a container and runs unchanged on any Docker host. Two
zero-cost managed targets are documented below.

---

## Option A — Streamlit Community Cloud (fastest)

Streamlit Cloud builds directly from the GitHub repo.

1. Push this repository to GitHub (see `README` / `git remote`).
2. Go to **https://share.streamlit.io** → *New app*.
3. Select the repo, branch **`main`**, and main file **`app/streamlit_app.py`**.
4. Deploy. The platform installs `requirements.txt`, the apt package in
   `packages.txt` (`libgomp1`, needed by XGBoost/LightGBM) and applies the theme
   from `.streamlit/config.toml`.

The app runs **offline by default** (`BMW_OFFLINE_MODE=true`), so it needs no
secrets. To enable live external APIs, add secrets under *App → Settings →
Secrets*.

---

## Option B — Hugging Face Spaces (Docker SDK)

A Space with the **Docker** SDK reuses this repo's `Dockerfile` verbatim.

### One-time, automated (CLI)

```bash
pip install -U huggingface_hub
hf auth login                      # paste a write token from hf.co/settings/tokens

# Create the Space and push the repo to it (run from the project root):
hf repo create bmw-sales-analytics --repo-type space --space-sdk docker -y
git remote add space https://huggingface.co/spaces/<your-username>/bmw-sales-analytics
git push space main
```

The Space's `README.md` must carry this front-matter for the Docker build to be
picked up (the project README already documents the app; for a Space, prepend or
use a dedicated README):

```yaml
---
title: BMW Luxury Sales Analytics
emoji: "◆"
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 8501
pinned: true
license: mit
---
```

> The image already runs as a non-root UID 1000 and listens on `0.0.0.0:8501`,
> which matches Hugging Face Spaces' runtime model.

---

## Option C — Any Docker host / self-managed

```bash
docker compose up --build          # -> http://localhost:8501
# or
docker build -t bmw-sales-analytics .
docker run -p 8501:8501 bmw-sales-analytics
```

A container health check on `/_stcore/health` is built in.

---

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `BMW_OFFLINE_MODE` | `true` | Use deterministic mock external data (no network/keys). Set `false` for live APIs. |
| `FX_API_KEY` | — | Optional exchangerate.host key (live FX). |
| `BMW_RANDOM_SEED` | `42` | Reproducibility. |
| `KMP_DUPLICATE_LIB_OK` | `TRUE` | Set in-image; avoids the Windows/Anaconda OpenMP clash for local dev. |
