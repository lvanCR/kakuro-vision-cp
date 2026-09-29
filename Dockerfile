FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib

# libglib2.0-0 y libgomp1: requeridas por OpenCV y PyTorch en Debian slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch para CPU (la CNN es pequeña; así la imagen funciona en cualquier equipo)
RUN pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

COPY src ./src
COPY models ./models
COPY data/examples ./data/examples

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
