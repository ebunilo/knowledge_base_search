FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Build tools are needed for some wheels on slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./pyproject.toml
COPY README.md ./README.md
COPY kb ./kb
COPY data ./data

RUN pip install --upgrade pip \
    && pip install .

EXPOSE 8765

CMD ["uvicorn", "kb.web.app:app", "--host", "0.0.0.0", "--port", "8765"]
