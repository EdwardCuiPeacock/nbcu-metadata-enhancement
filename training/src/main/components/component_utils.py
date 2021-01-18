import tensorflow as tf
import typing

_FEATURE_KEY = "features"
_LABEL_KEY = "series_ep_tags"
MAX_STRING_LENGTH = 277


def create_tag_lookup_table(tag_file):
    table = tf.lookup.StaticVocabularyTable(
        tf.lookup.TextFileInitializer(
            tag_file,
            key_dtype=tf.string,
            key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            value_dtype=tf.int64,
            value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
            delimiter=None,
        ),
        num_oov_buckets=1,
    )
    return table


def label_transform(x, y, num_tags, table):
    """Use the number of classes to convert the sparse tag indicies to dense"""
    # Need to add one for out-of-vocabulary tags in eval dataset
    # Then we drop this dimension so that we are only making predictions for in-vocab tags
    return (
        x,
        tf.cast(
            tf.sparse.to_indicator(table.lookup(y), vocab_size=num_tags + 1), tf.int32
        )[:, :-1],
    )


def _gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed fies"""
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def _input_fn(
    file_pattern,
    tf_transform_output,
    num_tags,
    table,
    batch_size=64,
    shuffle=True,
    epochs=None,
):
    """Generates features and label for tuning/training.
    Args:
        file_pattern: input tfrecord file pattern.
        tf_transform_output: A TFTransformOutput.
        batch_size: representing the number of consecutive elements of
          returned dataset to combine in a single batch
    Returns:
        A dataset that contains (features, indices) tuple where features
        is a dictionary of Tensors, and indices is a single Tensor of
        label indices.
    """
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        shuffle=shuffle,
        label_key=_LABEL_KEY,
        num_epochs=epochs,
    )
    return dataset.map(lambda x, y: label_transform(x, y, num_tags, table))
