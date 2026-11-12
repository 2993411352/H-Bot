FROM python:3.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" \
    NVIDIA_VISIBLE_DEVICES=void OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt && \
    python -c "import flask, av, pyarrow, numpy, PIL, waitress"
COPY . /app
ENTRYPOINT ["python", "/app/serve.py"]
