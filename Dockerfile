# --- Stage 1: Build the Wheel ---
FROM python:latest AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY . .

CMD "uv run train"