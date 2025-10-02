FROM python:3.13-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Финальная стадия
FROM python:3.13-slim

WORKDIR /app

# Устанавливаем системные зависимости для OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Копируем виртуальное окружение
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Копируем только необходимые файлы
COPY src/ ./src/
COPY data/ ./data/
COPY weights/ ./weights/

RUN mkdir -p results

RUN ls -la /app/weights/ && echo "Проверка файлов модели..."

WORKDIR /app/src

EXPOSE 8000

CMD ["uvicorn", "main:main_app", "--host", "0.0.0.0", "--port", "8000"]