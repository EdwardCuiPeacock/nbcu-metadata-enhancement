"""
TFX Pipeline for Peacock Deep Learning Recs Enginge model
"""
import os
import tensorflow_data_validation as tfdv

# Pipeline name will be used to identify this pipeline.
PIPELINE_NAME = "metadata-dev"

# Local testing data
DATA_PATH_TEST = "test_data/"
# Real BQ Data
with open('main/queries/ingest_query_test.sql', 'r') as input_query:
    query_test = input_query.read()
with open('main/queries/ingest_query.sql', 'r') as input_query:
    query = input_query.read()

IMAGE = "eu.gcr.io/ml-sandbox-101/custom_gpu_tfx_image_nbcu_3:local"

GCS_BUCKET_NAME = "metadata-bucket-sky"

GOOGLE_CLOUD_REGION = "europe-west2"
GOOGLE_CLOUD_PROJECT = "ml-sandbox-101"


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
TRAIN_NUM_STEPS = 100000
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


# # Use AI Platform training.
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

