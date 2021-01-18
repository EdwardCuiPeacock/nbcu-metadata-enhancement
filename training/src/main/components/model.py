import tensorflow as tf
import tensorflow_transform as tft
import pandas as pd
import numpy as np

from tensorflow.keras import callbacks, layers
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall

from google.cloud import storage
import pickle

import main.components.component_utils as component_utils
import main.pipelines.config as config

EMBEDDING_LOCATION = "gs://ml-sandbox-101-tagging/data/processed/training_data/glove_data/glove_embedding_index.pkl"


class AutoTaggingModel:
    def __init__(
        self,
        embedding_file,
        embedding_dim,
        train_embedding,
        output_size,
        vocab_size,
        vocab_df,
        max_string_length,
    ):
        self.__embedding_file = embedding_file
        self.__embedding_dim = embedding_dim
        self.__vocab_size = vocab_size
        self.__vocab_df = vocab_df
        self.__train_embedding = train_embedding
        self.__output_size = output_size
        self.__max_string_length = max_string_length

        self.__initialize_embedding_matrix()

    def __initialize_embedding_matrix(self):
        storage_client = storage.Client()

        # Better way to do this with os.path?
        split_path = self.__embedding_file.split("/")
        bucket_name = split_path[2]
        blob_name = ("/").join(split_path[3:])
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        pickle_in = blob.download_as_string()
        self.file = pickle.loads(pickle_in)

        self.embedding_matrix = np.zeros((self.__vocab_size, self.__embedding_dim))

        for i, word in enumerate(self.__vocab_df.values):
            embedding_vector = self.file.get(word[0])
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector

    def embedding_layer(self):
        return layers.Embedding(
            input_dim=self.__vocab_size,
            output_dim=self.__embedding_dim,
            weights=[self.embedding_matrix],
            input_length=self.__max_string_length,
            trainable=self.__train_embedding,
        )

    def n_grams_channel(self, inputs, n_words_filter: int):
        channel = layers.Conv2D(
            256, kernel_size=(n_words_filter, self.__embedding_dim), activation="relu"
        )(inputs)
        channel_mp = layers.MaxPool2D(pool_size=(channel.shape[1], 1))(channel)
        channel_final = layers.Flatten()(channel_mp)
        return channel_final

    def define_model(self):
        inputs = layers.Input(shape=(self.__max_string_length,), name="features")
        embedding = self.embedding_layer()(inputs)
        channel_inputs = layers.Reshape(
            target_shape=(self.__max_string_length, self.__embedding_dim, 1)
        )(embedding)
        channel1_final = self.n_grams_channel(channel_inputs, 3)
        channel2_final = self.n_grams_channel(channel_inputs, 4)
        channel3_final = self.n_grams_channel(channel_inputs, 5)
        channels_final = layers.Concatenate()(
            [channel1_final, channel2_final, channel3_final]
        )
        channels_final = layers.Dropout(rate=0.4)(channels_final)
        channels_final = layers.Dense(2000, "relu")(channels_final)
        predictions = layers.Dense(self.__output_size, "sigmoid")(channels_final)
        model = Model(inputs=inputs, outputs=predictions)

        return model

    def get_model(self):
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = self.define_model()

            metrics = [tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                loss=BinaryCrossentropy(),
                metrics=metrics,
            )
        return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(component_utils._LABEL_KEY)

        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

        transformed_features = model.tft_layer(parsed_features)

        outputs = model(transformed_features)
        return {"outputs": outputs}

    return serve_tf_examples_fn


def run_fn(fn_args):
    """Train the model based on given args

    Args:
        fn_args: Holds args used to train the model as name/value pairs
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    num_tags = tf_transform_output.vocabulary_size_by_name("tags")
    tag_file = tf_transform_output.vocabulary_file_by_name("tags")
    vocab_size = tf_transform_output.vocabulary_size_by_name("vocab")
    vocab_file = tf_transform_output.vocabulary_file_by_name("vocab")
    vocab_df = pd.read_csv(vocab_file, header=None)

    table = component_utils.create_tag_lookup_table(tag_file)

    train_dataset = component_utils._input_fn(
        file_pattern=fn_args.train_files,
        tf_transform_output=tf_transform_output,
        num_tags=num_tags,
        table=table,
        batch_size=config.BATCH_SIZE,
    )

    eval_dataset = component_utils._input_fn(
        file_pattern=fn_args.eval_files,
        tf_transform_output=tf_transform_output,
        batch_size=config.BATCH_SIZE,
        num_tags=num_tags,
        table=table,
    )

    model = AutoTaggingModel(
        embedding_dim=300,
        train_embedding=True,
        embedding_file=EMBEDDING_LOCATION,
        output_size=num_tags,  # Don't want to predict OOV-only training tags
        vocab_size=vocab_size + 1,
        vocab_df=vocab_df,
        max_string_length=component_utils.MAX_STRING_LENGTH,
    ).get_model()

    early_stopping_callback = callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=4,
        verbose=0,
        mode="auto",
        restore_best_weights=True,
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=2,
        verbose=0,
        mode="auto",
        min_delta=0.0001,
    )

    model.fit(
        train_dataset,
        epochs=config.EPOCHS,
        steps_per_epoch=fn_args.train_steps // config.EPOCHS,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[early_stopping_callback, reduce_lr],
    )

    signatures = {
        "serving_default": _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
    }

    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)
