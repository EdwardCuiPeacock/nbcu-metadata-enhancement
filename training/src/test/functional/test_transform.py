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

        self.initial_dataset = pd.read_parquet("test_data/parquet/train.parquet")

    def test_TFTTransformOutput(self):
        """
        Is the transform component outputting the expected files?
        """
        self.assertTrue(os.path.exists(self.transform_output))

        self.assertIsNotNone(self.tag_file)
        self.assertIsNotNone(self.vocab_file)

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
            tokenized_text = elem[0]["features_xf"]
            tags = elem[1]

        # Check that everything in batch matches the initial dataset
        # Where the initial dataset is original parquet file
        for i, tag in enumerate(tags):
            # Decode the multi-hot encoded label vector of tags
            tag_idx = np.argwhere(tag > 0).reshape(-1)
            decoded_tags = self.tag_df.iloc[tag_idx].values
            # Get the original tags as strings
            initial_tags = self.initial_dataset.iloc[i][1]
            self.assertTrue(sorted(initial_tags) == sorted(decoded_tags))

            # Create a new row for Out Of Vocabulary (OOV) instances
            self.vocab_df.loc[len(self.vocab_df)] = ""
            # Decode the tokenized and integerized text
            decoded_text = (" ").join(
                np.hstack(self.vocab_df.iloc[tokenized_text[i]].values)
            )
            # Get the original text as raw, uncleaned strings
            initial_text = self.initial_dataset.iloc[i][0]

            cleaned_initial = transform.clean(initial_text)
            self.assertTrue(
                decoded_text.strip() == cleaned_initial.numpy().decode("utf-8")
            )


if __name__ == "__main__":
    tf.test.main()
