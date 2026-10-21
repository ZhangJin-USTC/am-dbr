FROM python:3.12.14-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1 MPLBACKEND=Agg
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /work
COPY requirements-lock.txt .
RUN python -m pip install --no-cache-dir -r requirements-lock.txt
COPY . .
ENTRYPOINT ["python", "reproduce.py"]
CMD ["smoke"]
