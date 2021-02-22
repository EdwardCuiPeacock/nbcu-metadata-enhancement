"""
https://github.com/tensorflow/tfx/blob/master/tfx/examples/bigquery_ml/taxi_utils_bqml_test.py
"""

import tensorflow as tf
import numpy as np
import os
from mock import patch

import main.components.transform as transform

class TransformTest(tf.test.TestCase):
    def setUp(self):
        super(TransformTest, self).setUp()

    def test_transformed_name(self):
        names = "features"
        self.assertEqual("features_xf", transform._transformed_name(names))

    @patch("main.components.transform.LABEL_COLS", ['Animals', 'Anime'])
    def test_preprocessing_fn(self):
        inputs = {
            'program_longsynopsis': [["some input"], ["more input"]],
            'Animals': [[0], [1]],
            'Anime': [[0], [1]]
        }

        expected_output = {
            'program_longsynopsis_xf': tf.constant(["some input", "more input"]),
            'tags': tf.constant([[0, 0], [1, 1]])
        }
        # Have to examine each element individually
        outputs = transform.preprocessing_fn(inputs)

        self.assertAllEqual(outputs.keys(), expected_output.keys())
        self.assertAllEqual(outputs['program_longsynopsis_xf'], 
                            expected_output['program_longsynopsis_xf'])
        self.assertAllEqual(outputs['tags'], 
                            expected_output['tags'])

if __name__ == "__main__":
    tf.test.main()
