"""
TFX Pipeline for Peacock Deep Learning Recs Enginge model
"""
import os
import tensorflow_data_validation as tfdv
from jinja2 import Environment, FileSystemLoader
from functools import partial

# Pipeline name will be used to identify this pipeline.
PIPELINE_NAME = "metadata-dev"

IMAGE = "eu.gcr.io/ml-sandbox-101/custom_gpu_tfx_image_nbcu_4:local"

GCS_BUCKET_NAME = "metadata-bucket-sky"

GOOGLE_CLOUD_REGION = "europe-west2"
GOOGLE_CLOUD_PROJECT = "ml-sandbox-101"
BQ_DATASET = 'metadata_sky'
BQ_TABLE = 'merlin_movie_series_data_small'
TEST_LIMIT = 20 

# BQ data 
# TODO: This needs to go somewhere else
# TODO: Right now we are templating at pipeline COMPILE time 
#       eventually we'll want to do this at RUN time, since this 
#       will allow us to template out any temporal aspects of the query
file_loader = FileSystemLoader('main/queries')
env = Environment(loader=file_loader)
template = env.get_template('ingest_query.sql')

partially_rendered_query = partial(template.render, 
                                   project=GOOGLE_CLOUD_PROJECT, 
                                   dataset=BQ_DATASET, 
                                   table=BQ_TABLE)

query = partially_rendered_query()
query_test = partially_rendered_query(limit=TEST_LIMIT)

# Local testing data
DATA_PATH_TEST = "test_data/"


PREPROCESSING_FN = "main.components.transform.preprocessing_fn"
RUN_FN = "main.components.bert_model.run_fn"


# TODO: Should go somewhere else?
def get_domain_size(schema_path, feature):
    schema_text = tfdv.load_schema_text(schema_path)
    domain = tfdv.get_domain(schema_text, feature)

    return len(domain.value)

num_labels = get_domain_size('schema/schema.pbtxt', 'tags')
custom_config = {
    'num_labels': num_labels
}


# TODO: update this (too many steps?)
TRAIN_NUM_STEPS = 25000
EVAL_NUM_STEPS = 0

TRAIN_NUM_STEPS_TEST = 3 
EVAL_NUM_STEPS_TEST = 0


#############################
#### Infrastructure Configs #
#############################

# Beam args to use BigQueryExampleGen with Beam DirectRunner.
BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS = [
     "--project=" + GOOGLE_CLOUD_PROJECT,
     "--temp_location=" + os.path.join("gs://", GCS_BUCKET_NAME, "tmp"),
 ]


# Use AI Platform training.
GCP_AI_PLATFORM_TRAINING_ARGS = {
     "project": GOOGLE_CLOUD_PROJECT,
     "region": GOOGLE_CLOUD_REGION,
     "masterType": "n1-standard-16",
     "masterConfig": {
         "imageUri": IMAGE,
         "acceleratorConfig": {"count": 4, "type": "NVIDIA_TESLA_T4"},
     },
     "scaleTier": "CUSTOM",
 }

