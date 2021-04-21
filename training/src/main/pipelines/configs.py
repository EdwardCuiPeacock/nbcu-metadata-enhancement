"""
TFX Pipeline for Peacock Deep Learning Recs Enginge model
"""
import os
import tensorflow_data_validation as tfdv
import jinja2
from jinja2 import Environment, FileSystemLoader
from functools import partial

# Pipeline name will be used to identify this pipeline.
PIPELINE_NAME = "metadata_dev_edc_base_0_0_2" # TODO: change this

###IMPORTANT CHANGE THIS ALWAYS
MODEL_NAME = "ncr_meta_edc_dev_0_0_2" # TODO: change this, this is an entry on metadata_enhacement.model_results

GCS_BUCKET_NAME = "metadata-bucket-base" # TODO: HOLD

GOOGLE_CLOUD_REGION = "us-east1"
try:
    import google.auth  # pylint: disable=g-import-not-at-top
    try:
        _, GOOGLE_CLOUD_PROJECT = google.auth.default()
    except google.auth.exceptions.DefaultCredentialsError:
        GOOGLE_CLOUD_PROJECT = ''
except ImportError:
    GOOGLE_CLOUD_PROJECT = ''

OUTPUT_TABLE = "res-nbcupea-dev-ds-sandbox-001.metadata_enhancement.model_results"
GOOGLE_CLOUD_PROJECT = "res-nbcupea-dev-ds-sandbox-001"
BQ_DATASET = 'metadata_enhancement'
BQ_TABLE = 'merlin_data_with_lang_and_type'

TOKEN_LIMIT = 256

TEST_LIMIT = 20 

enable_cache = False

IMAGE = 'gcr.io/' + GOOGLE_CLOUD_PROJECT + '/edc-dev-pipeline'

# BQ data 
with open(os.path.join("main/queries/ingest_query_new_fixed.sql"), "r") as fid:
    query_str = fid.read() # read everything    
# Apply / parse any templated fields
query = jinja2.Template(query_str).render(
    GOOGLE_CLOUD_PROJECT=GOOGLE_CLOUD_PROJECT,
    TOKEN_LIMIT=TOKEN_LIMIT,
    TEST_LIMIT=None,)
# query_test = partially_rendered_query(limit=TEST_LIMIT)

# Local testing data
DATA_PATH_TEST = "test_data/" # TODO: src/

PREPROCESSING_FN = "main.components.transform.preprocessing_fn" # TODO: src.
RUN_FN = "main.components.bert_model.run_fn" # TODO: src.


# TODO: Should go somewhere else?
def get_domain_size(schema_path, feature):
    schema_text = tfdv.load_schema_text(schema_path)
    domain = tfdv.get_domain(schema_text, feature)

    return len(domain.value)

num_labels = get_domain_size('schema/schema.pbtxt', 'tags') # TODO: src/


## TRAINING ARGS
USE_STEPS = False
TRAIN_NUM_STEPS = 10000
EVAL_NUM_STEPS = 0

TRAIN_NUM_STEPS_TEST = 3 
EVAL_NUM_STEPS_TEST = 0

EPOCHS = 3
BATCH_SIZE = 128

custom_config = {
    'num_labels': num_labels,
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'use_steps': USE_STEPS,
    'seq_length': TOKEN_LIMIT,
}

#############################
#### Infrastructure Configs #
#############################

def set_memory_request_and_limits(memory_request, memory_limit):
    def _set_memory_request_and_limits(task):
        return (
            task.container.set_memory_request(memory_request)
                .set_memory_limit(memory_limit)
            )
        
    return _set_memory_request_and_limits

MEMORY_REQUEST = '10G'
MEMORY_LIMIT = '11G'

# Beam args to use BigQueryExampleGen with Beam DirectRunner.

BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS = [
     "--project=" + GOOGLE_CLOUD_PROJECT,
     "--temp_location=" + os.path.join("gs://", GCS_BUCKET_NAME, "tmp"),
     "--machine_type=n1-standard-16",
     "--disk_size_gb=100",
     "--runner=DataflowRunner",
     "--region=" + GOOGLE_CLOUD_REGION,
]


# Use AI Platform training.
GCP_AI_PLATFORM_TRAINING_ARGS = {
     "project": GOOGLE_CLOUD_PROJECT,
     "region": "us-central1",
     "masterType": "n1-highmem-16",
     "masterConfig": {
         "imageUri": IMAGE,
         "acceleratorConfig": {"count": 2, "type": "NVIDIA_TESLA_T4"}, #90ms per step with 4 GPUs
     },
     "scaleTier": "CUSTOM",
 }

