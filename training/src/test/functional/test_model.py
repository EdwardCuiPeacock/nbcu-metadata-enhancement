"""
https://github.com/tensorflow/tfx/blob/master/tfx/examples/bigquery_ml/taxi_utils_bqml_test.py

Here we will do some training tests to make sure that our model definition is correct and that it 
is able to learn on a small amount of data 
"""
import tensorflow as tf
from tensorflow.keras import callbacks

import os
import numpy as np

import main.components.model as model
import main.components.component_utils as component_utils

from test.functional.common_test_setup import CommonTestSetup


class TaggingModelTest(CommonTestSetup):
    def setUp(self):
        """
        Run the exact same setup as DataForTestingGenerator so we are looking in the same 
        place for pipeline artifacts

        Therefore, this MUST be run after generate_test_data 
        """
        super().setUp()
        super().setTransformOutputs()

    def _create_model(self, embedding_dim, train_embedding, max_string_length):
        """
        Initialise the model with the correct parameters
        """
        tagging_model_uncompiled = model.AutoTaggingModel(
            embedding_dim=embedding_dim,
            train_embedding=train_embedding,
            embedding_file=model.EMBEDDING_LOCATION,
            output_size=self.num_tags,
            vocab_size=self.vocab_size + 1,
            vocab_df=self.vocab_df,
            max_string_length=max_string_length,
        )

        return tagging_model_uncompiled

    def test_model(self):
        tagging_model_uncompiled = self._create_model(
            300, True, component_utils.MAX_STRING_LENGTH
        )
        tagging_model = tagging_model_uncompiled.get_model()

        # Test the embedding matrix is correctly initialized by
        # calling the embedding layer and see if it gives expected results
        # Embedding is always after the input layer
        embedding_layer = tagging_model.layers[1]
        # Some german text we know is in the data
        words = ["und", "angriff", "ein", "amerikaner"]
        tokenized_words = self.vocab_lookup_table.lookup(tf.constant(words))
        # Look up the embeddings manually from the file and compare to calling
        # embedding layer
        embeddings = np.vstack(
            [tagging_model_uncompiled.file.get(word) for word in words]
        )
        self.assertAllEqual(embeddings, embedding_layer.call(tokenized_words))

        # Unknown words should get mapped to zeros
        unknown_word = ["lkjsdljd", "", "exercise"]
        tokenized_unknown_word = self.vocab_lookup_table.lookup(
            tf.constant(unknown_word)
        )
        unknown_embedding = embedding_layer.call(tf.constant(tokenized_unknown_word))
        self.assertEqual(tf.math.count_nonzero(unknown_embedding), 0)

        # Get the initial weights
        all_initial_weights = [weights.numpy() for weights in tagging_model.weights]

        early_stopping_callback = callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.0001,
            patience=4,
            verbose=0,
            mode="auto",
            restore_best_weights=True,
        )

        history = tagging_model.fit(
            self.transformed_train_dataset,
            epochs=10,
            steps_per_epoch=1,
            validation_data=self.transformed_eval_dataset,
            validation_steps=1,
            callbacks=[early_stopping_callback],
        )

        # Make sure we have the expected metrics 
        all_metrics = tagging_model.metrics
        self.assertIsInstance(all_metrics[1], tf.keras.metrics.Precision)
        self.assertIsInstance(all_metrics[2], tf.keras.metrics.Recall)

        # Don't want the losses to be not-a-number
        self.assertFalse(tf.math.is_nan(history.history["val_loss"][-1]))
        self.assertFalse(tf.math.is_nan(history.history["loss"][-1]))

        # Is loss going down? We should be able to overfit on a small number of 
        # examples.
        self.assertLess(history.history["loss"][-1], history.history["loss"][0])
        
        # Get the final weights after fitting
        all_final_weights = [weights.numpy() for weights in tagging_model.weights]

        for initial, final in zip(all_initial_weights, all_final_weights):
            # Weights should be updated
            self.assertFalse(np.all(initial == final))
            # Weights are not more than half zero (arbitrary threshold). Point is 
            # to make sure that we are not getting a lot of dead neurons 
            self.assertGreater(tf.math.count_nonzero(final) / final.size, 0.5)
            # Don't want any weights to be not a number. Would mostly likely indicate 
            # that gradients are exploding
            num_nans = tf.reduce_sum(tf.cast(tf.math.is_nan(final), tf.int32))
            self.assertEqual(num_nans / final.size, 0)

        # Make sure we can evaluate
        _ = tagging_model.evaluate(self.transformed_eval_dataset, steps=1)

        # Make sure we can predict and that predictions are in range 
        # We are using sigmoid activation, so they should be in range (0, 1)
        predictions = tagging_model.predict(self.transformed_eval_dataset, steps=1)

        self.assertTrue(np.all((predictions > 0) & (predictions < 1)))


if __name__ == "__main__":
    tf.test.main()
