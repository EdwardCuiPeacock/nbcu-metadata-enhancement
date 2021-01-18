from mock import patch

import tensorflow as tf
import pandas as pd
import numpy as np

import main.components.model as model


def create_model(param_overrides={}):
    """Helper function to create an instance of the AutoTaggingModel,
    with the ability to override the default parameters"""
    default_params = {
        "embedding_file": "file://bucket/blob1/blob2",
        "embedding_dim": 300,
        "train_embedding": True,
        "output_size": 1000,
        "vocab_size": 100,
        "vocab_df": pd.DataFrame(["first", "second", "third"]),
        "max_string_length": 200,
    }
    params = default_params.copy()
    params.update(param_overrides)

    return model.AutoTaggingModel(**params)


class AutoTaggingModelTest(tf.test.TestCase):
    def setUp(self):
        self.patch_client = patch("main.components.model.storage.Client", autospec=True)
        self.patch_pickle_loads = patch(
            "main.components.model.pickle.loads", autospec=True
        )
        self.mock_client = self.patch_client.start()
        self.mock_pickle_loads = self.patch_pickle_loads.start()
        self.addCleanup(self.patch_client.stop)
        self.addCleanup(self.patch_pickle_loads.stop)

        self.mock_pickle_loads.return_value = {
            "first": np.zeros(300),
            "second": np.zeros(300),
            "third": np.zeros(300),
        }

    def test_autotagging_model_initializes_embedding_matrix(self):
        tagging_model = create_model()
        bucket = self.mock_client().bucket
        bucket.assert_called_with("bucket")
        blob = bucket("").blob
        blob.assert_called_with("blob1/blob2")

        download_as_string = blob("").download_as_string
        download_as_string.assert_called()

        self.assertEqual(tagging_model.embedding_matrix.shape, (100, 300))

    def test_embedding_layer(self):
        tagging_model = create_model()
        embedding_layer = tagging_model.embedding_layer()
        expected_layer = tf.keras.layers.Embedding(
            input_dim=100,
            output_dim=300,
            weights=tagging_model.embedding_matrix,
            input_length=200,
            trainable=True,
        )
        self.assertAllEqual(embedding_layer.weights, expected_layer.weights)
        self.assertEqual(embedding_layer.input_length, expected_layer.input_length)
        self.assertEqual(embedding_layer.trainable, expected_layer.trainable)

    def test_embedding_layer_is_not_trainable(self):
        tagging_model = create_model({"train_embedding": False})
        embedding_layer = tagging_model.embedding_layer()
        expected_layer = tf.keras.layers.Embedding(
            input_dim=100,
            output_dim=300,
            weights=tagging_model.embedding_matrix,
            input_length=200,
            trainable=False,
        )
        self.assertEqual(embedding_layer.trainable, expected_layer.trainable)

    def test_n_grams_channel(self):
        tagging_model = create_model()
        inputs = np.ones((32, 200, 300, 1))
        channel = tagging_model.n_grams_channel(inputs, 3)
        self.assertEqual(channel.shape, (32, 256))

        channel_with_different_n_words_filter = tagging_model.n_grams_channel(inputs, 5)
        self.assertEqual(channel_with_different_n_words_filter.shape, (32, 256))

    def test_define_model(self):
        tagging_model = create_model()
        model_definition = tagging_model.define_model()
        self.assertEqual(model_definition.input_names, ["features"])
        self.assertEqual(model_definition.input_shape, (None, 200))
        self.assertEqual(model_definition.output_names, ["dense_1"])
        self.assertEqual(model_definition.output_shape, (None, 1000))
        self.assertFalse(model_definition._is_compiled)

        # Testing Intermediate Shapes
        channel_1_output_shape = model_definition.layers[3].output.shape
        channel_2_output_shape = model_definition.layers[4].output.shape
        channel_3_output_shape = model_definition.layers[5].output.shape

        # Manually calculate output shape of convolutions based on kernel size of channel
        self.assertAllEqual(channel_1_output_shape, (None, 200 - 3 + 1, 1, 256))
        self.assertAllEqual(channel_2_output_shape, (None, 200 - 4 + 1, 1, 256))
        self.assertAllEqual(channel_3_output_shape, (None, 200 - 5 + 1, 1, 256))

    def test_get_model(self):
        tagging_model = create_model()
        model = tagging_model.get_model()

        self.assertTrue(model._is_compiled)

        self.assertIsInstance(model.distribute_strategy, tf.distribute.MirroredStrategy)

        expected_optimizer_config = tf.keras.optimizers.Adam().get_config().copy()
        expected_optimizer_config.update({"learning_rate": 0.0005})
        self.assertEqual(model.optimizer.get_config(), expected_optimizer_config)

        self.assertIsInstance(model.loss, tf.keras.losses.BinaryCrossentropy)


if __name__ == "__main__":
    tf.test.main()
