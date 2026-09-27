# --- Stage 1: Build the Wheel ---
FROM python:latest

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY ./pyproject.toml ./pyproject.toml
COPY ./uv.lock ./uv.lock
COPY ./README.md ./README.md
RUN uv sync