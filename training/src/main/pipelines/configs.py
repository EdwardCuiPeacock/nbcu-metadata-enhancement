"""
TFX Pipeline for Peacock Deep Learning Recs Enginge model
"""
import os

# Pipeline name will be used to identify this pipeline.
PIPELINE_NAME = "metadata-dev"

# TODO: Won't need this once we're certain BQ works 
DATA_PATH_TEST = "gs://metadata-bucket-sky/series_data_split_small/"
DATA_PATH = "gs://metadata-bucket-sky/series_data/"

IMAGE = "eu.gcr.io/ml-sandbox-101/custom_gpu_tfx_image_nbcu_1:local"

GCS_BUCKET_NAME = "metadata-bucket-sky"

GOOGLE_CLOUD_REGION = "europe-west2"
GOOGLE_CLOUD_PROJECT = "ml-sandbox-101"


PREPROCESSING_FN = "main.components.transform.preprocessing_fn"
RUN_FN = "main.components.bert_model.run_fn"

# TODO: update this (too many steps?)
TRAIN_NUM_STEPS = 2000
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
         "acceleratorConfig": {"count": 1, "type": "NVIDIA_TESLA_T4"},
     },
     "scaleTier": "CUSTOM",
 }

