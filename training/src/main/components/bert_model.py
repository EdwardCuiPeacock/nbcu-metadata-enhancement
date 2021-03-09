import tensorflow as tf
import tensorflow_transform as tft
import pandas as pd
import numpy as np

from tensorflow.keras import callbacks, layers

from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow_text
import tensorflow_hub as hub
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall

# TODO: Add these in config instead of hard-coding
TFHUB_HANDLE_PREPROCESSOR = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
TFHUB_HANDLE_ENCODER = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1"

def _gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed fies"""
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")

def _input_fn(file_pattern, tf_transform_output, batch_size=64, shuffle=True, epochs=None):
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
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        shuffle=shuffle,
        label_key='labels_xf',
        num_epochs=epochs
    )
    return dataset

def build_bert_tagger(num_labels):
    # TODO: think about alternative architecture
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='synopsis_xf')
    preprocessing_layer = hub.KerasLayer(TFHUB_HANDLE_PREPROCESSOR, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(TFHUB_HANDLE_ENCODER, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    output = tf.keras.layers.Dense(num_labels, activation="sigmoid")(net)
    return tf.keras.Model(text_input, output)

def get_compiled_model(num_labels):
    # TODO: figure out more about optimizer 
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = build_bert_tagger(num_labels)
        metrics = [tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        # TODO: clipnorm only seems to work in TF 2.4? 
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00003,
                                               clipnorm=1,
                                               epsilon=1e-8),
            loss=BinaryCrossentropy(),
            metrics=metrics,
        )
    return model

def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses JSON input"""
    # TODO: Create alternative serving function, especially if using evaluator
    model.tft_layer = tf_transform_output.transform_features_layer()
    
    @tf.function
    def serve_tf_examples_fn(raw_text):
        """Returns the output to be used in the serving signature."""
        reshaped_text = tf.reshape(raw_text, [-1, 1])
        transformed_features = model.tft_layer({"synopsis": reshaped_text})

        outputs = model(transformed_features)
        return {"outputs": outputs}

    return serve_tf_examples_fn

def run_fn(fn_args):
    """Train the model based on given args
    
    Args:
        fn_args: Holds args used to train the model as name/value pairs
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    # Not sure why its like this
    # TODO: fix this, might be a version issue?
    num_labels = fn_args.custom_config['num_labels']
    
    train_dataset = _input_fn(
        file_pattern=fn_args.train_files,
        tf_transform_output=tf_transform_output,
        batch_size=64)
    
    model = get_compiled_model(num_labels)
    
    # TODO pass in epochs
    history = model.fit(
        train_dataset, 
        epochs=3,
        steps_per_epoch=fn_args.train_steps // 3
    )
    
    signatures = {
        "serving_default": _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
    }
    
    model.save(
        fn_args.serving_model_dir, save_format="tf", signatures=signatures
    )



