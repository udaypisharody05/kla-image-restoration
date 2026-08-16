FROM python:3.12.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app \
    && mkdir -p /app/data /app/results \
    && chown -R app:app /app

# All root-level scripts (train.py, inference.py, evaluate_checkpoint.py,
# benchmark_inference.py, export_final_weights.py, etc.) plus the library
# code, test suite, and the small tracked final-weights artifact. Excludes
# the dataset, .venv, checkpoints/, and restored_test_outputs/ via
# .dockerignore.
COPY --chown=app:app *.py pytest.ini ./
COPY --chown=app:app src/ ./src/
COPY --chown=app:app tests/ ./tests/
COPY --chown=app:app weights/ ./weights/

USER app

CMD ["python", "-m", "pytest", "-m", "not integration", "-q"]
