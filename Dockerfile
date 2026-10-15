
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV STOCKFISH_PATH=/usr/local/bin/stockfish

WORKDIR /app

# Install the Linux version of Stockfish
RUN apt-get update \
    && apt-get install -y --no-install-recommends stockfish \
    && ln -sf "$(command -v stockfish)" /usr/local/bin/stockfish \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code and the YOLO model
COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
