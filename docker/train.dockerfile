FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# Install Poetry
RUN pip install poetry

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the project files (pyproject.toml and poetry.lock) into the working directory
COPY pyproject.toml poetry.lock ./

# Install project dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction --no-ansi

COPY src/ /usr/src/app/src/
COPY data/ /usr/src/app/data/

ENTRYPOINT ["python", "src/main.py"]