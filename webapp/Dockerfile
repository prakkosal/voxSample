FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install build deps only for the pip install step, then drop them.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt \
 && apt-get purge -y --auto-remove build-essential

COPY main.py /app/main.py
COPY static /app/static

EXPOSE 8000

ENV HOST=0.0.0.0 \
    PORT=8000

CMD ["python", "main.py"]
