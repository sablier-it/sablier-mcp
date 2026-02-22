FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir .

ENV MCP_TRANSPORT=streamable-http
ENV MCP_PORT=8000

EXPOSE 8000

CMD ["sablier-mcp"]
