FROM tensorflow/tfx:0.26.0
#WORKDIR /pipeline
#COPY ./ ./
ENV PYTHONPATH="/pipeline:${PYTHONPATH}"
RUN python -m pip install --upgrade poetry
RUN pip install tensorflow_text

# This can just be avoied if we simply build the pipeline within src/, 
# rather than outside of src/
COPY src src 
WORKDIR /src