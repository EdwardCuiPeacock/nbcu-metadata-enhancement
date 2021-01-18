"""
This file contains the preprocessing_fn callback function for the transform
component in the tfx pipeline
"""
import tensorflow_transform as tft
import tensorflow as tf

import main.components.component_utils as component_utils

# Filter out everything other than German characters
regex_filter = "[^äÄöÖüÜßa-zA-Z]"


def _transformed_name(key):
    return key + "_xf"


def clean(text):
    """
    Clean up text input by removing everything but German words

    Args:
        text: tensors containing the synopsis. (batch_size/None, 1)

    Returns:
        final: Cleaned text
    """
    # Encoding needed to keep german characters
    lower = tf.strings.lower(text, "utf-8")
    cleaned = tf.strings.regex_replace(lower, regex_filter, " ")
    # Filter single letters
    single_letters_removed = tf.strings.regex_replace(
        cleaned, r"((^|\s)[äÄöÖüÜßa-zA-Z]{1})+\s", " "
    )
    stripped = tf.strings.strip(single_letters_removed)
    # Replace multiple spaces with single space
    final = tf.strings.regex_replace(stripped, " +", " ")

    return final


def cleaned_to_padded(cleaned_text):
    """
    Take cleaned text and turn it into a padded tensor of appropriate size

    Args:
        cleaned_text: tensor containing a string

    Returns:
        text_tokens: Tokenized text padded to be same length
    """
    # Ragged tensor since features will have different lengths, this allows us to deal
    # with batched examples
    text_tokens = tf.compat.v1.string_split(
        cleaned_text, " ", result_type="RaggedTensor"
    )
    # Since shape is specified, result will be padded and/or truncated to the specified shape
    # This means we don't need to worry about text that is either too short or too long
    text_tokens = text_tokens.to_tensor(shape=[None, component_utils.MAX_STRING_LENGTH])

    return text_tokens


def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs

    Args:
        inputs: map from feature keys to raw not-yet-transformed-features

    Returns:
        Map from string feature key to transformed feature operations
    """
    outputs = {}

    text = tf.squeeze(inputs[component_utils._FEATURE_KEY], axis=1)
    tags = inputs[component_utils._LABEL_KEY]

    cleaned = clean(text)
    text_tokens = cleaned_to_padded(cleaned)
    text_indices = tft.compute_and_apply_vocabulary(
        text_tokens, vocab_filename="vocab", num_oov_buckets=1
    )

    # compute vocab of tags, do not apply due to serving issues
    _ = tft.vocabulary(tags, vocab_filename="tags")

    outputs[_transformed_name(component_utils._FEATURE_KEY)] = text_indices
    outputs[component_utils._LABEL_KEY] = tags

    return outputs
