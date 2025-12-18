FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy==2.2.6

# Установка PyTorch CPU
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    torchvision==0.17.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html
# Установка остальных зависимостей
COPY req.txt .
RUN pip install --no-cache-dir -r req.txt

# Копирование приложения и модели
COPY main.py .
COPY model.pth .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
