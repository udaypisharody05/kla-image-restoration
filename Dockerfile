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

COPY --chown=app:app inspect_dataset.py visualize_samples.py evaluate_baseline.py pytest.ini ./
COPY --chown=app:app src/ ./src/
COPY --chown=app:app tests/ ./tests/

USER app

CMD ["python", "-m", "pytest", "-m", "not integration", "-q"]
