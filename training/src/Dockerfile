FROM tensorflow/tfx:0.26.0
WORKDIR /pipeline
COPY ./ ./
ENV PYTHONPATH="/pipeline:${PYTHONPATH}"
RUN pip install tensorflow_text
RUN pip install pyarrow==3.0.0
RUN pip install google-cloud-storage==1.32.0 
RUN pip install google-cloud-bigquery-storage==2.4.0
RUN pip install gcsfs==0.8.0
