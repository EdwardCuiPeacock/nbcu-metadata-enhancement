import tensorflow as tf
import tensorflow_transform as tft
import pandas as pd
import numpy as np

from tensorflow.keras import callbacks, layers

from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow_addons as tfa
import tensorflow_text  # Registers the ops for preprocessing
import tensorflow_hub as hub
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall

from main.components.tagger_model import TaggerModel

def _gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed fies"""
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def _input_fn(
    file_pattern, tf_transform_output, batch_size=64, shuffle=True, epochs=None
):
    """Generates features and label for tuning/training.
    Args:
        file_pattern: input tfrecord file pattern.
        tf_transform_output: A TFTransformOutput.
        batch_size: representing the number of consecutive elements of
          returned dataset to combine in a single batch
    Returns:
        A dataset that contains (features, indices) tuple where features
        is a dictionary of Tensors, and indices is a single Tensor of
        label indices.
    """
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        shuffle=shuffle,
        label_key="tags_xf",
        num_epochs=epochs,
    )

    print("print what the data looks like")
    for ii in dataset:
        print(ii)
        break

    return dataset

def get_compiled_model(num_labels, seq_length):
    # TODO: figure out more about optimizer
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = TaggerModel(num_labels, seq_length)
        metrics = [
            tf.keras.metrics.BinaryAccuracy(threshold=0.5),
            #"kullback_leibler_divergence",
            "cosine_similarity",
            tfa.metrics.F1Score(num_classes=num_labels, threshold=0.5, average="macro"),
            #tf.keras.metrics.AUC(curve="ROC", name="ROC_AUC"),
            tf.keras.metrics.AUC(curve="PR", multi_label=True, name="pr_auc"),
        ]        
        loss_func = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25, gamma=3.0)
        print("Loss function")
        print(loss_func.__dict__)
        model.compile(
            optimizer="adam",
            loss=loss_func,
            metrics=metrics,
        )
    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses JSON input"""
    # TODO: Create alternative serving function, especially if using evaluator
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(raw_text, raw_keywords):
        """Returns the output to be used in the serving signature."""
        reshaped_text = tf.reshape(raw_text, [-1, 1])
        reshaped_keywords = tf.reshape(raw_keywords, [-1, 1])
        transformed_features = model.tft_layer(
            {"synopsis": reshaped_text, 
             "keywords": reshaped_keywords,
            })

        outputs = model(transformed_features)
        return {"outputs": outputs}

    return serve_tf_examples_fn


def run_fn(fn_args):
    """Train the model based on given args

    Args:
        fn_args: Holds args used to train the model as name/value pairs
    """
    print("Fn args")
    print(fn_args)
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs available: {gpus}")

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    # Not sure why its like this
    # TODO: fix this, might be a version issue?
    num_labels = fn_args.custom_config["num_labels"]
    num_epochs = fn_args.custom_config["epochs"]
    batch_size = fn_args.custom_config["batch_size"]
    seq_length = fn_args.custom_config["seq_length"]
    print(f"Num labels: {num_labels}")

    model = get_compiled_model(num_labels, seq_length)

    if fn_args.custom_config["use_steps"]:

        train_dataset = _input_fn(
            file_pattern=fn_args.train_files,
            tf_transform_output=tf_transform_output,
            batch_size=batch_size,
        )

        history = model.fit(
            train_dataset,
            epochs=num_epochs,
            steps_per_epoch=fn_args.train_steps // num_epochs,
            verbose=1,
        )
    else:
        train_dataset = _input_fn(
            file_pattern=fn_args.train_files,
            tf_transform_output=tf_transform_output,
            batch_size=batch_size,
            epochs=1,
        )

        # Find out how large the dataset is
        count_rows = 0
        for kk in train_dataset:
            count_rows += kk[-1].shape[0]
        print(f"Total number of rows of training: {count_rows}")

        history = model.fit(train_dataset, epochs=num_epochs, verbose=1)

    # raise(ValueError("Artificial Error: attempt to rerun the model"))

    signatures = {
        "serving_default": _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="synopsis"),
            tf.TensorSpec(shape=[None], dtype=tf.string, name="keywords"), # titles
            #tf.SparseTensorSpec(shape=[None, None], dtype=tf.string), # keywords
        ),
    }

    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)
