# TFX build 
FROM tensorflow/tfx:0.27.0 as tfx_base

# Main 
FROM tensorflow/tensorflow:2.4.0-gpu 
COPY --from=tfx_base tfx/src/ tfx-src 

RUN python -m pip install --upgrade poetry 
COPY pyproject.toml .
RUN poetry config virtualenvs.create false
RUN poetry install 

COPY src src 
WORKDIR /src 

# RUN pytest test/unit/

#RUN pip install google-api-python-client
#RUN pip install oauth2client

ENV PYTHONPATH="/src:${PYTHONPATH}"
ENV PYTHONPATH="/tfx-src:${PYTHONPATH}"

ENTRYPOINT ["python3.6", "/tfx-src/tfx/scripts/run_executor.py"]
