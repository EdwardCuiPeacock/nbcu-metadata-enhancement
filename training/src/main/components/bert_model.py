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

# TODO: Add these in config instead of hard-coding
TFHUB_HANDLE_PREPROCESSOR = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
TFHUB_HANDLE_ENCODER = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1"
#TFHUB_HANDLE_ENCODER = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/2"
TOKEN_EMBEDDINGS = "gs://edc-dev/kubeflowpipelines-default/tfx_pipeline_output/node2vec_sports_syn_0_1_1/Trainer/model/19130/serving_model_dir"

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


def build_bert_tagger_old(num_labels, seq_length):
    # TODO: think about alternative architecture
    preprocessor = hub.load(TFHUB_HANDLE_PREPROCESSOR)
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="synopsis")
    tokenize = hub.KerasLayer(preprocessor.tokenize, name="tokenize")
    tokenized_inputs = [tokenize(text_input)]
    preprocessing_layer = hub.KerasLayer(
        preprocessor.bert_pack_inputs,
        arguments=dict(seq_length=seq_length),
        name="preprocessing",
    )
    # preprocessing_layer = hub.KerasLayer(TFHUB_HANDLE_PREPROCESSOR, name='preprocessing')
    encoder_inputs = preprocessing_layer(tokenized_inputs)
    # TODO: try freezing the BERT encoder layer
    encoder = hub.KerasLayer(TFHUB_HANDLE_ENCODER, trainable=False, name="BERT_encoder")
    outputs = encoder(encoder_inputs)
    net = outputs["pooled_output"]
    # Outputs
    hidden1 = tf.keras.layers.Dense(512, activation="relu")(net)
    drop1 = tf.keras.layers.Dropout(0.2)(hidden1)
    hidden2 = tf.keras.layers.Dense(256, activation="relu")(drop1)
    drop2 = tf.keras.layers.Dropout(0.2)(hidden2)
    output = tf.keras.layers.Dense(num_labels, activation="sigmoid")(drop2)
    model = tf.keras.Model(text_input, output)
    print(model.summary())
    return model


def build_bert_tagger(num_labels, seq_length):
    model = TaggerModel(TFHUB_HANDLE_PREPROCESSOR, TFHUB_HANDLE_ENCODER, 
        TOKEN_EMBEDDINGS, num_labels, seq_length)
    return model


def get_compiled_model(num_labels, seq_length):
    # TODO: figure out more about optimizer
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = build_bert_tagger(num_labels, seq_length)
        metrics = [
            "accuracy",
            #"kullback_leibler_divergence",
            "cosine_similarity",
            #tf.keras.metrics.AUC(curve="ROC", name="ROC_AUC"),
            #tf.keras.metrics.AUC(curve="PR", name="PR_AUC"),
        ]
        # clipnorm only seems to work in TF 2.4 with distribution strategy
        def cos_sim(y_true, y_pred, axis=1):
            y_true = tf.cast(y_true, tf.float64)
            y_pred = tf.cast(y_pred, tf.float64)
            return tf.keras.losses.cosine_similarity(y_true, y_pred, axis=axis)
        
        def focal_loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            return tfa.losses.sigmoid_focal_crossentropy(y_true, y_pred, alpha=0.75, gamma=3.0)

        model.compile(
            optimizer="adam",
            loss=focal_loss,
            metrics=metrics,
        )
    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses JSON input"""
    # TODO: Create alternative serving function, especially if using evaluator
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(raw_text, tokens): # keywords
        """Returns the output to be used in the serving signature."""
        reshaped_text = tf.reshape(raw_text, [-1, 1])
        transformed_features = model.tft_layer(
            {"synopsis": reshaped_text, 
            "tokens": tokens, 
            #'keywords': keywords
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
    max_token_length = fn_args.custom_config["max_token_length"]
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
            tf.SparseTensorSpec(shape=[None, None], dtype=tf.string), # token
            #tf.SparseTensorSpec(shape=[None, None], dtype=tf.string), # keywords
        ),
    }

    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)
