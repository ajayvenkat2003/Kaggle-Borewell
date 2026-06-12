FROM python:latest
RUN pip install uv
COPY . .
ENTRYPOINT "test.sh"

