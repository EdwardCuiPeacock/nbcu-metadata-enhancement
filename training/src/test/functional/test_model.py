"""
https://github.com/tensorflow/tfx/blob/master/tfx/examples/bigquery_ml/taxi_utils_bqml_test.py

Here we will do some training tests to make sure that our model definition is correct and that it
is able to learn on a small amount of data
"""
import tensorflow as tf
from tensorflow.keras import callbacks

import os
import numpy as np

import main.components.bert_model as bert_model

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

    def test_model(self):
        # TODO: Do this in configs or something?
        num_labels = 408
        model = bert_model.get_compiled_model(num_labels)

        # Don't use encoder and dense layer now, maybe something we can do with them? 
        preprocessing_layer, encoder_layer, dense_layer = model.layers[1:]

        # No weights in preprocessing_layer 
        self.assertEqual(len(preprocessing_layer.weights), 0)

        # Get the initial weights
        all_initial_weights = [weights.numpy() for weights in model.weights]

        history = model.fit(
            self.transformed_train_dataset,
            epochs=10,
            steps_per_epoch=1,
        )
        # Don't want the losses to be not-a-number
        self.assertFalse(tf.math.is_nan(history.history["loss"][-1]))

        # Is loss going down? We should be able to overfit on a small number of
        # examples.
        self.assertLess(history.history["loss"][-1], history.history["loss"][0])

        # Get the final weights after fitting
        all_final_weights = [weights.numpy() for weights in model.weights]

        for initial, final in zip(all_initial_weights, all_final_weights):
            # Some weights are just "true" or "false"
            # TODO: honestly not sure what these weights are, should figure that out 
            if type(initial) is np.bool_:
                continue

            self.assertFalse(np.all(initial == final))
            # Weights are not more than half zero (arbitrary threshold). Point is
            # to make sure that we are not getting a lot of dead neurons
            self.assertGreater(tf.math.count_nonzero(final) / final.size, 0.5)
            # Don't want any weights to be not a number. Would mostly likely indicate
            # that gradients are exploding
            num_nans = tf.reduce_sum(tf.cast(tf.math.is_nan(final), tf.int32))
            self.assertEqual(num_nans / final.size, 0)

        # Make sure we can evaluate
        _ = model.evaluate(self.transformed_train_dataset, steps=1)

        # Make sure we can predict and that predictions are in range
        # We are using sigmoid activation, so they should be in range (0, 1)
        predictions = model.predict(self.transformed_train_dataset, steps=1)

        self.assertTrue(np.all((predictions > 0) & (predictions < 1)))


if __name__ == "__main__":
    tf.test.main()
