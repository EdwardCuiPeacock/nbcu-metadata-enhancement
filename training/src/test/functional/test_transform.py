import tensorflow as tf

import os
import pandas as pd
import numpy as np

import main.components.transform as transform

from test.functional.common_test_setup import CommonTestSetup


class TransformTest(CommonTestSetup):
    def setUp(self):
        """
        Run the exact same setup as DataForTestingGenerator so we are looking in the same
        place for pipeline artifacts

        Therefore, this MUST be run after generate_test_data
        """
        super().setUp()
        super().setTransformOutputs()

        self.initial_dataset = pd.read_parquet("test_data/test_data.parquet")

    def test_TFTTransformOutput(self):
        """
        Is the transform component outputting the expected files?
        """
        self.assertTrue(os.path.exists(self.transform_output))

        self.assertIsNotNone(self.label_file)
        self.assertIsNotNone(self.tag_file)

    def test_expected_transform_behavior(self):
        """
        If we use the output assets from transform (vocab files),
        can we decode the dataset we create from the transform examples and retrieve
        the initial dataset values?

        We want to ensure that transform component and our utils for creating
        a dataset are doing exactlywhat we are expecting them to do
        """
        # Here we are getting all of the transformed examples
        # One batch is 10 elements, which is the number of test examples
        for elem in self.transformed_train_dataset.take(1):
            text = elem[0]["synopsis"]
            labels = elem[1]

        # Check that everything in batch matches the initial dataset
        # Where the initial dataset is original parquet file
        for i, label in enumerate(labels):
            # Decode the multi-hot encoded label vector of labels
            label_idx = np.argwhere(label > 0).reshape(-1)
            decoded_labels = self.label_df.iloc[label_idx].values
            # Get the original tags as strings
            initial_labels = self.initial_dataset['labels'].iloc[i]
            self.assertTrue(sorted(initial_labels) == sorted(decoded_labels))

            # "decoded text" should be unchanged from original
            decoded_text = text[i]
            initial_text = self.initial_dataset['synopsis'].iloc[i]
            self.assertTrue(decoded_text == tf.constant(initial_text))

        # Check the tag vocabulary 
        raw_tags = np.unique(np.hstack(self.initial_dataset['tags'].values))
        tag_vocab = np.hstack(self.tag_df.values)

        self.assertTrue(sorted(raw_tags) == sorted(tag_vocab))


if __name__ == "__main__":
    tf.test.main()
