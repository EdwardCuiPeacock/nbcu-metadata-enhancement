# TFX build 
FROM tensorflow/tfx:0.27.0 as tfx_base

# Main 
FROM tensorflow/tensorflow:2.4.1-gpu 
COPY --from=tfx_base tfx/src/ tfx-src 

WORKDIR /pipeline
COPY ./ ./
RUN pip install tensorflow_text
RUN pip install pyarrow==3.0.0
RUN pip install google-cloud-storage==1.32.0 
RUN pip install google-cloud-bigquery-storage==2.4.0

ENV PYTHONPATH="/src:${PYTHONPATH}"
ENV PYTHONPATH="/tfx-src:${PYTHONPATH}"

ENTRYPOINT ["python3.7", "/tfx-src/tfx/scripts/run_executor.py"]





