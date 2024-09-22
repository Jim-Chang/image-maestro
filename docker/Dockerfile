# builder stage
FROM python:3.11.7-slim-bookworm AS builder

RUN pip install --upgrade pip && \
    pip install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app
COPY poetry.lock pyproject.toml /app/

RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR


# runtime stage
FROM python:3.11.7-slim-bookworm AS runtime

EXPOSE 8000

RUN useradd -ms /bin/bash maestro
USER maestro

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app

WORKDIR /app

COPY --chown=maestro --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY --chown=maestro . /app

CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
