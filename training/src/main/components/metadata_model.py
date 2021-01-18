from google.cloud import bigquery
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import sys
from collections import OrderedDict
from datetime import datetime
from metadata_model.model import bert_model
import time
import json

# from google.cloud.storage import blob
import tensorflow_transform as tft
from absl import logging
from typing import Text, List

BATCH_SIZE = 256
max_seq_len = 50
n_genres = 32


def transformed_name(key: Text) -> Text:
    """Generate the name of the transformed feature from original name."""
    return key


tag_index = dict(
    enumerate(
        [
            "Action & Adventure",
            "Animated",
            "Anime",
            "Biography",
            "Children's/Family Entertainment",
            "Comedy",
            "Courtroom",
            "Crime",
            "Documentary",
            "Drama",
            "Educational",
            "Fantasy",
            "Gay and Lesbian",
            "History",
            "Holiday",
            "Horror",
            "Martial arts",
            "Military & War",
            "Music",
            "Musical",
            "Mystery",
            "Romance",
            "Science fiction",
            "Sports",
            "Thriller",
            "Western",
            "kids (ages 5-9)",
            "not for kids",
            "older teens (ages 15+)",
            "preschoolers (ages 2-4)",
            "teens (ages 13-14)",
            "tweens (ages 10-12)",
        ]
    )
)


def _build_bert_model():

    model = bert_model.BertEmbeddingModel(
        max_seq_len,
        n_genres,
        {"learning_rate": 0.00003, "clipnorm": 1, "epsilon": 1e-8},
        {"threshold": 0.5, "name": "macro_f1", "average": "macro"},
        tag_index=tag_index,
        data_name="merlin_movies_kids_buckets_clean",
    )

    model.setup()

    return model


def _gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed files."""
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example and applies TFT."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

        transformed_features = model.tft_layer(parsed_features)
        transformed_features.pop(transformed_name("label"), None)

        return model(transformed_features)

    return serve_tf_examples_fn


def _input_fn(file_pattern, tf_transform_output, batch_size=BATCH_SIZE):
    """Generates features and label for tuning/training.

    Args:
      file_pattern: input tfrecord file pattern.
      tf_transform_output: A TFTransformOutput.
      batch_size: representing the number of consecutive elements of returned
        dataset to combine in a single batch

    Returns:
      A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=BATCH_SIZE,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key=transformed_name("label"),
    )

    return dataset


def run_fn(fn_args):
    """Train the model based on given args.
    Args:
        fn_args: Holds args used to train the model as name/value pairs.
    """

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, BATCH_SIZE)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, BATCH_SIZE)

    print("#######", tf.config.experimental.list_physical_devices("GPU"))
    # mirrored_strategy = tf.distribute.MirroredStrategy() ### commented as not used?

    model = _build_bert_model()
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq="batch"
    )

    model.compile()
    model.model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        batch_size=BATCH_SIZE,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback],
    )

    signatures = {
        "serving_default": _get_serve_tf_examples_fn(
            model.model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
    }
    model.model.save(fn_args.serving_model_dir, signatures=signatures, save_format="tf")
    model.save_to_path(fn_args.serving_model_dir)
