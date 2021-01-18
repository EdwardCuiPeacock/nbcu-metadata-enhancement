"""
TFX Pipeline for Peacock Deep Learning Recs Enginge model
"""

import os


# Pipeline name will be used to identify this pipeline.
# PIPELINE_NAME = 'metadata_enhancement5'
PIPELINE_NAME = "metadata-dev"
# DATA_PATH = "gs://pkds-datascience/media_genome/movies_parquet/"
DATA_PATH = "gs://metadata-dev-bucket/tok_data"

##############################
#### GCP related configs  ####
##############################
try:
    import google.auth  # pylint: disable=g-import-not-at-top

    try:
        _, GOOGLE_CLOUD_PROJECT = google.auth.default()
    except google.auth.exceptions.DefaultCredentialsError:
        GOOGLE_CLOUD_PROJECT = ""
except ImportError:
    GOOGLE_CLOUD_PROJECT = ""

# GCS_BUCKET_NAME = GOOGLE_CLOUD_PROJECT + '-kubeflowpipelines-default'
GCS_BUCKET_NAME = "metadata-dev-bucket"
DATA_PATH = "gs://metadata-dev-bucket/tok_data"

GCP_SDK_IMAGE_URI = (
    "gcr.io/google.com/cloudsdktool/cloud-sdk:latest"  # 'google/cloud-sdk:278.0.0'
)

GOOGLE_CLOUD_REGION = "us-east1"

#############################
#### Execution Configs  ####
#############################

PREPROCESSING_SCRIPT = """
mkdir code
gsutil cp -r gs://metadata-dev-bucket/metadata_enhancements code
cd code/metadata_enhancements/pipelines
apt-get update
apt-get install python3.7
echo "INSTALLING RECS"
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
pip install pandas
pip install matplotlib
pip install tensorflow==2.3.1
pip install transformers
pip install tensorflow-addons
pip install google-cloud-storage
pip install google-cloud-bigquery
pip install google-cloud-bigquery-storage
pip install pyarrow
cd pipeline
echo "RUNNING TOKENIZATION"
export FNAME='token_meta'
python3 preprocess.py $FNAME
echo "COPYING BACK"
gsutil cp ${FNAME}.parquet gs://metadata-dev-bucket/tok_data/
gsutil cp ${FNAME}_with_ids.parquet gs://metadata-dev-bucket/tok_data/

"""
REC_EVAL_SCRIPT = """"""
PREPROCESSING_FN = "metadata_model.preprocessing.preprocessing_fn"
RUN_FN = "metadata_model.metadata_model.run_fn"
# RUN_FN = 'models.estimator.model.run_fn'

TRAIN_NUM_STEPS = int((210162 / 256) * 5)
EVAL_NUM_STEPS = 760

# Change this value according to your use cases.
EVAL_ACCURACY_THRESHOLD = 0.6


#############################
#### Infrastructure Configs #
#############################

# Beam args to use BigQueryExampleGen with Beam DirectRunner.
BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS = [
    "--project=" + GOOGLE_CLOUD_PROJECT,
    "--temp_location=" + os.path.join("gs://", GCS_BUCKET_NAME, "tmp"),
]


DATAFLOW_BEAM_PIPELINE_ARGS = [
    "--project=" + GOOGLE_CLOUD_PROJECT,
    "--runner=DataflowRunner",
    "--temp_location=" + os.path.join("gs://", GCS_BUCKET_NAME, "tmp"),
    "--region=" + GOOGLE_CLOUD_REGION,
    "--experiments=shuffle_mode=auto",
    "--disk_size_gb=50",
]

# Use AI Platform training.
GCP_AI_PLATFORM_TRAINING_ARGS = {
    "project": GOOGLE_CLOUD_PROJECT,
    "region": GOOGLE_CLOUD_REGION,
    "masterType": "n1-highmem-16",
    "masterConfig": {
        "imageUri": "gcr.io/"
        + GOOGLE_CLOUD_PROJECT
        + "/peacock-tfx-metadata-dev-pipeline",
        "acceleratorConfig": {"count": 1, "type": "NVIDIA_TESLA_T4"},
    },
    "scaleTier": "CUSTOM",
}

# A dict which contains the serving job parameters to be passed to Google
# Cloud AI Platform. For the full set of parameters supported by Google Cloud AI
# Platform, refer to
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.models
# TODO(step 9): (Optional) Uncomment below to use AI Platform serving.
GCP_AI_PLATFORM_SERVING_ARGS = {
    "model_name": PIPELINE_NAME,
    "project_id": GOOGLE_CLOUD_PROJECT,
    # The region to use when serving the model. See available regions here:
    # https://cloud.google.com/ml-engine/docs/regions
    # Note that serving currently only supports a single region:
    # https://cloud.google.com/ml-engine/reference/rest/v1/projects.models#Model  # pylint: disable=line-too-long
    "regions": [GOOGLE_CLOUD_REGION],
    "machine-type": "n1-standard-4",
}
