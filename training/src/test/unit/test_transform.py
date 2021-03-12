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

    def test_binarize_tags(self):
        sparse_tags = tf.sparse.SparseTensor(indices=[[0,0], [1,1]], 
                                            values=[1.0, 2.0], 
                                            dense_shape=[2, 3])

        binarized_tags = transform.binarize_tags(sparse_tags, 4)
        self.assertAllEqual(binarized_tags, 
                            tf.constant([[0, 1, 0, 0], 
                                        [0, 0, 1, 0]], dtype=tf.int64))
    
    def test_binarize_tags_fails_when_out_of_bounds(self):
        sparse_tags = tf.sparse.SparseTensor(indices=[[0,0], [1,1]], 
                                            values=[1.0, 5.0], 
                                            dense_shape=[2, 3])

        with self.assertRaises(tf.errors.InvalidArgumentError):
            binarized_tags = transform.binarize_tags(sparse_tags, 4)


    @patch("main.components.transform.tft.vocabulary", autospec=True)
    @patch("main.components.transform.tft.compute_and_apply_vocabulary", autospec=True)
    def test_preprocessing_fn(self, mockComputeAndApply, mockVocabulary):
        inputs = {
            'synopsis': [["some input"], ["more input"]],
            'tags': [['tag_1'], ['tag_2']]
        }
        custom_config = {
            'num_labels': 5
        }
        mockComputeAndApply.return_value = tf.sparse.SparseTensor(
            indices=[[0, 0], [1, 0]],
            values=[1, 2],
            dense_shape=[2, 3]
        )
        expected_output = {
            'synopsis': tf.constant(["some input", "more input"]),
            'tags_xf': tf.constant([[0, 1, 0, 0, 0],
                                      [0, 0, 1, 0, 0]], dtype=tf.int64)
        }

        # Have to examine each element individually
        outputs = transform.preprocessing_fn(inputs, custom_config)

        call = mockComputeAndApply.call_args_list
        self.assertEqual(call[0][1], {"vocab_filename": "tags", "num_oov_buckets": 1})
        self.assertAllEqual(outputs.keys(), expected_output.keys())
        self.assertAllEqual(outputs['synopsis'], 
                            expected_output['synopsis'])
        self.assertAllEqual(outputs['tags_xf'], 
                            expected_output['tags_xf'])

if __name__ == "__main__":
    tf.test.main()
