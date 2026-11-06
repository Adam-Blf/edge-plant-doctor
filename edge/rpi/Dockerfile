# Plant Doctor edge image for Raspberry Pi (linux/arm64) and any other arm64 board.
# Build from the repository root:
#   docker buildx build --platform linux/arm64 -f edge/rpi/Dockerfile -t plant-doctor-edge .
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLANT_DOCTOR_HOME=/app \
    PYTHONPATH=/app/src:/app/edge/rpi

WORKDIR /app

COPY edge/rpi/requirements.txt edge/rpi/requirements.txt
RUN pip install --no-cache-dir -r edge/rpi/requirements.txt

# Only the pieces inference needs: no torch, no training code.
COPY src/edgeplant/__init__.py src/edgeplant/config.py src/edgeplant/preprocess.py \
     src/edgeplant/latency.py src/edgeplant/labels.py src/edgeplant/
COPY edge/rpi/plantdoctor_edge edge/rpi/plantdoctor_edge
COPY models/plant-doctor-int8.onnx models/labels.json models/

RUN useradd --system --uid 10001 --no-create-home plant
USER 10001

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/health', timeout=3)"

# The service refuses a non-loopback bind without PLANT_DOCTOR_TOKEN.
CMD ["python", "-m", "plantdoctor_edge.server", "--host", "0.0.0.0", "--port", "8080"]
