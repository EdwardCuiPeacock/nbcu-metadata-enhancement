FROM python:3.7-slim

WORKDIR /app

RUN apt-get update
RUN apt-get install -y git

# Install pre-commit
RUN pip install --upgrade pre-commit

COPY . /app


RUN pre-commit install
RUN pre-commit run --all-files
