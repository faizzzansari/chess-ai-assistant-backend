
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV STOCKFISH_PATH=/usr/games/stockfish
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

WORKDIR /app

# Stockfish and Linux libraries used by OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       stockfish \
       libxcb1 \
       libglib2.0-0 \
       libgl1 \
    && test -x /usr/games/stockfish \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
