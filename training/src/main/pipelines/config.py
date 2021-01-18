"""
Example config file 
Here you can put constants, file paths, GCS buckets, etc.
"""

GCS_BUCKET_NAME = "gcs-somewhere"
GOOGLE_CLOUD_REGION = "europe-west2"

PREPROCESSING_FN = "main.components.transform.preprocessing_fn"
PIPELINE_NAME = "my-little-pipeline"

# Based on Dataset sizes in preprocessing.ipynb notebook
TRAIN_DATASET_SIZE = 90543
EVAL_DATASET_SIZE = 8549
