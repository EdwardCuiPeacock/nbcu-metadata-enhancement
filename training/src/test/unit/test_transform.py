"""
https://github.com/tensorflow/tfx/blob/master/tfx/examples/bigquery_ml/taxi_utils_bqml_test.py
"""

import tensorflow as tf
import numpy as np
from tensorflow_transform import beam as tft_beam
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import schema_utils
import apache_beam as beam
import json

from tensorflow_transform.beam import tft_unit

import os

import main.components.transform as transform
import main.components.component_utils as component_utils

from mock import patch


class TransformTest(tf.test.TestCase):
    def setUp(self):
        super(TransformTest, self).setUp()

    def test_transformed_name(self):
        names = "features"
        self.assertEqual("features_xf", transform._transformed_name(names))

    def test_regex_filter(self):
        text_to_keep = "äÄüÜöÖßWelkjaBsa"
        text_to_filter = "£$% ~()[]é45"

        self.assertNotRegex(text_to_keep, transform.regex_filter)
        self.assertRegex(text_to_filter, transform.regex_filter)

    def test_clean_converts_feature_to_lower_case(self):
        cleaned_text = "upper case words"
        self.assertEqual(cleaned_text, transform.clean("UPPER CASE WORDS"))

    def test_clean_removes_punctuation(self):
        cleaned_text = "what boudicca ridiculous"
        self.assertEqual(
            cleaned_text, transform.clean("""what?! 'Boudicca"?! #ridiculous:""")
        )

    def test_clean_replaces_punctuation_with_spaces(self):
        cleaned_text = "split with punctuation that needs immediate removal right"
        self.assertEqual(
            cleaned_text,
            transform.clean(
                "split.with,punctuation(that)needs\\immediate/removal-right"
            ),
        )

    def test_clean_removes_preceding_and_proceeding_whitespace(self):
        cleaned_text = "hello"
        self.assertEqual(cleaned_text, transform.clean("   hello   "))

    def test_clean_replaces_newline_with_spaces(self):
        cleaned_text = "eyyup denice"
        self.assertEqual(cleaned_text, transform.clean("eyyup\n\ndenice"))

    def test_clean_removes_extra_spaces(self):
        cleaned_text = "look dees spaces bruv"
        self.assertEqual(
            cleaned_text, transform.clean("look     dees  \n\t    spaces bruv")
        )

    def test_clean_removes_single_letters(self):
        cleaned_text = "this is clean text"
        self.assertEqual(cleaned_text, transform.clean("this is-a-clean text"))

    def test_clean_to_padded_handles_short_text(self):
        cleaned_text = tf.constant(["eyyup denice", "hello"])
        text_tokens = transform.cleaned_to_padded(cleaned_text)

        self.assertEqual(
            tuple(text_tokens.shape), (2, component_utils.MAX_STRING_LENGTH)
        )
        # Check the first string
        self.assertTrue(np.all(text_tokens[0][2:] == ""))
        self.assertTrue(np.all(text_tokens[0][:2] == tf.constant(["eyyup", "denice"])))

        # Check the second string
        self.assertTrue(np.all(text_tokens[1][1:] == ""))
        self.assertEqual(text_tokens[1][0], tf.constant(["hello"]))

    def test_clean_to_padded_handles_long_text(self):
        # First element longer, second shorter than max string length
        cleaned_text = tf.constant([("a " * 300).strip(), ("b " * 150).strip()])
        text_tokens = transform.cleaned_to_padded(cleaned_text)

        self.assertEqual(
            tuple(text_tokens.shape), (2, component_utils.MAX_STRING_LENGTH)
        )

        self.assertTrue(np.all(text_tokens[0] == "a"))
        self.assertTrue(np.all(text_tokens[1][:150] == "b"))
        self.assertTrue(np.all(text_tokens[1][150:] == ""))

    @patch("main.components.component_utils.MAX_STRING_LENGTH", 4)
    @patch("main.components.transform.tft.vocabulary", autospec=True)
    @patch("main.components.transform.tft.compute_and_apply_vocabulary", autospec=True)
    def test_preprocessing_fn(self, mockComputeAndApply, mockVocabulary):
        inputs = {
            component_utils._FEATURE_KEY: [["some input"], ["more input"]],
            component_utils._LABEL_KEY: [["tag_1"], ["tag_1", "tag_2"]],
        }

        mockComputeAndApply.return_value = ["transformed_text"]
        outputs = transform.preprocessing_fn(inputs)

        expected_tokens = tf.convert_to_tensor(
            [["some", "input", "", ""], ["more", "input", "", ""]], dtype=tf.string
        )

        # Assert we called it correctly, need to do weird stuff b/c np array
        # See https://stackoverflow.com/questions/56644729/mock-assert-mock-calls-with-a-numpy-array-as-argument-raises-valueerror-and-np
        call = mockComputeAndApply.call_args_list
        self.assertTrue(np.all(expected_tokens == call[0][0]))
        self.assertEqual(call[0][1], {"vocab_filename": "vocab", "num_oov_buckets": 1})

        mockVocabulary.assert_called_with(
            inputs[component_utils._LABEL_KEY], vocab_filename="tags"
        )

        self.assertAllEqual(
            outputs[component_utils._LABEL_KEY], inputs[component_utils._LABEL_KEY]
        )
        self.assertAllEqual(outputs["features_xf"], ["transformed_text"])


if __name__ == "__main__":
    tf.test.main()
