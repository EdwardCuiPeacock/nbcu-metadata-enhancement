import tensorflow as tf
import main.components.component_utils as component_utils
from mock import patch
from unittest.mock import Mock
import numpy as np


class TestComponentUtils(tf.test.TestCase):
    @patch(
        "main.components.component_utils.tf.lookup.StaticVocabularyTable", autospec=True
    )
    @patch(
        "main.components.component_utils.tf.lookup.TextFileInitializer", autospec=True
    )
    def test_create_tag_lookup_table(self, mockInitializer, mockTable):
        tag_file = "afilesomewhere"
        initializer = "initializer"
        mockInitializer.return_value = initializer
        mockTable.return_value = "table"
        table = component_utils.create_tag_lookup_table(tag_file)
        mockInitializer.assert_called_with(
            tag_file,
            key_dtype=tf.string,
            key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            value_dtype=tf.int64,
            value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
            delimiter=None,
        )
        mockTable.assert_called_with(initializer, num_oov_buckets=1)
        self.assertEqual(table, "table")

    @patch("main.components.component_utils.tf.sparse.to_indicator")
    def test_label_transform(self, mockSparseToIndicator):
        x = [0, 1, 2]
        y = "fake tag"
        num_tags = 5
        lookup_table = Mock()
        lookup_table.lookup.return_value = [3.0, 2.0]
        mockSparseToIndicator.return_value = [[2.0, 3.0], [2.0, 3.0]]
        transform_x, transform_y = component_utils.label_transform(
            x, y, num_tags, lookup_table
        )
        lookup_table.lookup.assert_called_with("fake tag")
        mockSparseToIndicator.assert_called_with([3.0, 2.0], vocab_size=6)
        self.assertEqual(x, transform_x)
        self.assertEqual(transform_y.dtype, tf.int32)

    @patch("main.components.component_utils.tf.data.TFRecordDataset", autospec=True)
    def test_gzip_reader_fn(self, mockDataset):
        filenames = tf.data.Dataset.from_tensor_slices(["file1", "file2", "file3"])

        mockDataset.return_value = "did we return dataset?"
        tfrecord = component_utils._gzip_reader_fn(filenames)

        mockDataset.assert_called_with(filenames, compression_type="GZIP")
        self.assertEqual(tfrecord, "did we return dataset?")

    @patch("main.components.component_utils.label_transform")
    @patch(
        "main.components.component_utils.tf.data.experimental.make_batched_features_dataset",
        autospec=True,
    )
    def test_input_fn_uses_defaults(self, mockBatchedDataset, mockLabelTransform):
        file_pattern = "path/to/file/*"
        tf_transform_output_mock = Mock()
        feature_spec = {"feature": tf.io.VarLenFeature(tf.int32)}
        tf_transform_output_mock.transformed_feature_spec.return_value.copy.return_value = (
            feature_spec
        )
        num_tags = 5
        table = "table"

        test_dataset = tf.data.Dataset.from_tensor_slices((["features"], ["tags"]))

        mockBatchedDataset.return_value = test_dataset
        mockLabelTransform.return_value = "transformed labels"
        dataset = component_utils._input_fn(
            file_pattern, tf_transform_output_mock, num_tags, table
        )

        mockBatchedDataset.assert_called_with(
            file_pattern=file_pattern,
            batch_size=64,
            features=feature_spec,
            reader=component_utils._gzip_reader_fn,
            shuffle=True,
            label_key="series_ep_tags",
            num_epochs=None,
        )

        self.assertEqual(mockLabelTransform.call_count, 1)
        for elem in dataset.take(1):
            self.assertEqual(elem, "transformed labels")

    @patch("main.components.component_utils.label_transform")
    @patch(
        "main.components.component_utils.tf.data.experimental.make_batched_features_dataset",
        autospec=True,
    )
    def test_input_fn_overrides_defaults(self, mockBatchedDataset, mockLabelTransform):
        file_pattern = "path/to/file/*"
        tf_transform_output_mock = Mock()
        feature_spec = {"feature": tf.io.VarLenFeature(tf.int32)}
        tf_transform_output_mock.transformed_feature_spec.return_value.copy.return_value = (
            feature_spec
        )
        num_tags = 5
        table = "table"
        batch_size = 32
        shuffle = False
        epochs = 10

        test_dataset = tf.data.Dataset.from_tensor_slices(
            (["features", "features_2"], ["tags", "tags_2"])
        )

        mockBatchedDataset.return_value = test_dataset
        mockLabelTransform.return_value = "transformed labels 2"

        dataset = component_utils._input_fn(
            file_pattern,
            tf_transform_output_mock,
            num_tags,
            table,
            batch_size=batch_size,
            shuffle=shuffle,
            epochs=epochs,
        )
        mockBatchedDataset.assert_called_with(
            file_pattern=file_pattern,
            batch_size=32,
            features=feature_spec,
            reader=component_utils._gzip_reader_fn,
            shuffle=False,
            label_key="series_ep_tags",
            num_epochs=10,
        )

        self.assertEqual(mockLabelTransform.call_count, 1)
        for elem in dataset.take(2):
            self.assertEqual(elem, "transformed labels 2")


if __name__ == "__main__":
    tf.test.main()
