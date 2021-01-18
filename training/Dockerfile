# Testing build
FROM python:3.6.9-slim as test_base

RUN pip install poetry

COPY training/pyproject.toml .
RUN poetry config virtualenvs.create false
RUN poetry install

COPY training/src src
WORKDIR src

RUN pytest test/unit/

# Create dependencies
RUN poetry export -f requirements.txt --without-hashes --output requirements.generate.txt

# TFX build
FROM tensorflow/tfx:0.24.1 as tfx_base

# Main
FROM tensorflow/tensorflow:2.3.1-gpu
COPY --from=tfx_base tfx-src/ tfx-src

RUN python -m pip install --upgrade pip

# Take requirements.generate.txt
COPY --from=test_base requirements.generate.txt .
RUN python -m pip install -r requirements.generate.txt

COPY --from=test_base src src
WORKDIR /src
ENV PYTHONPATH="/src:${PYTHONPATH}"
ENV PYTHONPATH="/tfx-src:${PYTHONPATH}"

ENTRYPOINT ["python3.6", "/tfx-src/tfx/scripts/run_executor.py"]
