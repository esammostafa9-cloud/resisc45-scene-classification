# REST API image for the scene classifier.
#
#   docker build -t resisc45-api .
#   docker run -p 8000:8000 -v "$PWD/models:/models" resisc45-api
#
# The model is not baked into the image. Mount a folder with model.keras + class_names.json
# at /models, or leave RESISC45_MODEL_PATH unset to download from the Hugging Face Hub.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    RESISC45_MODEL_PATH=/models/model.keras

WORKDIR /app

# Dependencies first, so code changes don't invalidate the (large) TensorFlow layer.
COPY requirements.txt requirements-api.txt ./
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt -r requirements-api.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "resisc45.api:app", "--host", "0.0.0.0", "--port", "8000"]
