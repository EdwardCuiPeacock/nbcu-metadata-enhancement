# TFX build 
FROM tensorflow/tfx:0.27.0 as tfx_base

# Main 
FROM tensorflow/tensorflow:2.4.1-gpu 
COPY --from=tfx_base tfx/src/ tfx-src

RUN python -m pip install --upgrade poetry 
COPY pyproject.toml .
RUN poetry config virtualenvs.create false
RUN poetry install 

WORKDIR /pipeline
COPY ./ ./
#RUN pip install tensorflow_text==2.4.3
#RUN pip install ml_metadata==0.27.0
#RUN pip install pyarrow==3.0.0
#RUN pip install google-cloud-storage==1.32.0 
#RUN pip install google-cloud-bigquery-storage==2.4.0

# ENV PYTHONPATH="/src:${PYTHONPATH}"
ENV PYTHONPATH="/tfx-src:${PYTHONPATH}"

ENTRYPOINT ["python3.7", "/tfx-src/tfx/scripts/run_executor.py"]





