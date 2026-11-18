FROM python:3.12-slim
RUN python -m pip install --no-cache-dir numpy==2.1.3 pandas==2.2.3
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
USER 65534:65534
