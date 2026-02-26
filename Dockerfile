FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir .

ENV MCP_TRANSPORT=streamable-http

# Cloud Run sets PORT automatically (default 8080)
EXPOSE 8080

CMD ["sablier-mcp"]
