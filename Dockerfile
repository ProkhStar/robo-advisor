# Dockerfile para correr a Streamlit app (frontend)
FROM python:3.11-slim

# variáveis de ambiente para evitar prompts
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# copiar requirements e instalar
COPY requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && apt-get remove -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# copiar código
COPY . /app

EXPOSE 8501

# comando para correr a app
CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.port=8501", "--server.headless=true"]
