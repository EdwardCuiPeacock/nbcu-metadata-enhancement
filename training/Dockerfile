FROM tensorflow/tfx:0.26.0
#WORKDIR /pipeline
#COPY ./ ./
ENV PYTHONPATH="/pipeline:${PYTHONPATH}"
RUN python -m pip install --upgrade poetry
RUN pip install tensorflow_text

COPY src src 
WORKDIR /src