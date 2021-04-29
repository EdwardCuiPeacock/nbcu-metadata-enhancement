"""
For now this component mostly just passes through the data. 
One bit of processing we currently do is to join together all 
of the labels. Once (if) we move this into the query at the front 
of the pipeline, we might be able to get rid of this component 
entirely 
"""
import tensorflow_transform as tft
import tensorflow as tf

FEATURE = 'synopsis'
LABEL = 'tags'
TOKENS = 'tokens'
KEYWORDS = 'keywords'

def _transformed_name(key):
    return key + '_xf'

def compute_tags(transformed_tags, num_labels): 
    """
    Function for turning tags from sparse tensor to multilabel binarized 
    data. The final result is that tags is a binary matrix with shape (none, NUM_TAGS) 
    indicating the presence of a tag in an example
    Args: 
        transformed_tags: sparse tensor with transformed tags
        Looks like: 
            [[1, 3, 2, 6, x, x, x],
             [2, 3, 0, x, x, x, x],
             [3, 0, 5, 2, 4, 1, 6]]
    Returns: 
        Binarized tags, tensor with shape (none, NUM_TAGS)
        Looks like 
            [[0, 1, 1, 1, 0, 0, 0],
             [1, 0, 1, 1, 0, 0, 0],
             [1, 1, 1, 1, 1, 1, 1]]
        Given vocab size of 7
    """
    tags_multi_binarized = tf.sparse.to_indicator(transformed_tags, 
                                                  vocab_size=num_labels)
    #tags_multi_binarized = tf.cast(tags_multi_binarized, tf.float32)
    # Normalize the tags by their sum
    #tags_normalized = tags_multi_binarized / tf.reduce_sum(tags_multi_binarized, axis=1, keepdims=True)
    #return tags_normalized 
    return tf.cast(tags_multi_binarized, tf.int64)


def compute_tokens(tokens, max_token_length):
    """Convert a sparse tensor to RaggedTensor."""
    tokens = tf.sparse.reorder(tokens)
    # Ragged tensor is taking care of a lot of caveats
    # 1) With values unmapped to the vocab_list, the default value is set -1
    # when re-reading again during training, the out of vocab will be removed
    # 2) If we have more tokens at inference time, setting the shape will
    # remove any extra tokens, still making sure the maximum number of token
    # during training is preserved
    out = tf.RaggedTensor.from_value_rowids(values=tokens.values, \
                                           value_rowids=tokens.indices[:, 0])
    out = out.to_tensor(default_value=-1, shape=(None, max_token_length))
    out = tf.sort(out, axis=1, direction="DESCENDING")
    return out

def preprocessing_fn(inputs, custom_config):
    """Preprocess input columns into transformed columns."""
    outputs = {}
    text = tf.squeeze(inputs[FEATURE], axis=1)
    labels = inputs[LABEL]
    tokens = inputs[TOKENS]
    #keywords = inputs[KEYWORDS]    

    num_labels = custom_config.get('num_labels')
    
    # Create and apply a full vocabulary for the labels (subgenres)
    labels = tft.compute_and_apply_vocabulary(
       labels, vocab_filename=LABEL, num_oov_buckets=1
    )
    # labels = tft.apply_vocabulary(labels, 
    #     deferred_vocab_filename_tensor=tf.constant("gs://metadata-bucket-base/tfx-metadata-dev-pipeline-output/metadata_dev_edc_base_0_0_3/Transform/transform_graph/20890/transform_fn/assets/tags"),
    #     num_oov_buckets=1,
    #     )
    vocab_file = tf.constant(custom_config["token_vocab_list"])
    tokens = tft.apply_vocabulary(tokens, 
       deferred_vocab_filename_tensor=vocab_file,
       num_oov_buckets=0)

    # keywords = tft.apply_vocabulary(keywords,
    #     deferred_vocab_filename_tensor=vocab_file,
    #     num_oov_buckets=0)

    outputs[FEATURE] = text
    outputs[_transformed_name(LABEL)] = compute_tags(labels, num_labels)
    outputs[TOKENS] = compute_tokens(tokens, custom_config["max_token_length"])
    #outputs[KEYWORDS] = compute_tokens(keywords, custom_config["max_keyword_length"])
    
    return outputs

