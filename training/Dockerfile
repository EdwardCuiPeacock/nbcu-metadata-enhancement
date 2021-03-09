# TFX build 
FROM tensorflow/tfx:0.26.0 as tfx_base

# Main 
FROM tensorflow/tensorflow:2.3.1-gpu 
COPY --from=tfx_base tfx/src/ tfx-src 

RUN python -m pip install --upgrade poetry 

COPY training/pyproject.toml .
RUN poetry config virtualenvs.create false
RUN poetry install 

COPY training/src src 
WORKDIR /src 

# RUN pytest test/unit/

ENV PYTHONPATH="/src:${PYTHONPATH}"
ENV PYTHONPATH="/tfx-src:${PYTHONPATH}"

ENTRYPOINT ["python3.6", "/tfx-src/tfx/scripts/run_executor.py"]
