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

def _transformed_name(key):
    return key + '_xf'

def compute_tags(transformed_tags, num_labels): 
    """
    Function for turning tags from sparse tensor to multilabel binarized 
    data. The final result is that tags is a binary matrix with shape (none, NUM_TAGS) 
    indicating the presence of a tag in an example
    Args: 
        transformed_tags: sparse tensor with transformed tags 
    Returns: 
        Binarized tags, tensor with shape (none, NUM_TAGS)
    """
    tags_multi_binarized = tf.sparse.to_indicator(transformed_tags, 
                                                  vocab_size=num_labels)
    #tags_multi_binarized = tf.cast(tags_multi_binarized, tf.float32)
    # Normalize the tags by their sum
    #tags_normalized = tags_multi_binarized / tf.reduce_sum(tags_multi_binarized, axis=1, keepdims=True)
    #return tags_normalized 
    return tf.cast(tags_multi_binarized, tf.int64)


def compute_tokens(tokens):
    """Convert a sparse tensor to RaggedTensor."""
    tokens = tf.sparse.reorder(tokens)
    out = tf.RaggedTensor.from_value_rowids(values=tokens.indices[:, 1], \
                                            value_rowids=tokens.indices[:, 0])
    return out

def preprocessing_fn(inputs, custom_config):
    """Preprocess input columns into transformed columns."""
    outputs = {}
    text = tf.squeeze(inputs[FEATURE], axis=1)
    labels = inputs[LABEL]
    tokens = inputs[TOKENS]
    
    num_labels = custom_config.get('num_labels')
    
    # Create and apply a full vocabulary for the labels (subgenres)
    labels = tft.compute_and_apply_vocabulary(
       labels, vocab_filename=LABEL, num_oov_buckets=1
    )
    # labels = tft.apply_vocabulary(labels, 
    #     deferred_vocab_filename_tensor=tf.constant("gs://metadata-bucket-base/tfx-metadata-dev-pipeline-output/metadata_dev_edc_base_0_0_3/Transform/transform_graph/20890/transform_fn/assets/tags"),
    #     num_oov_buckets=1,
    #     )

    tokens = tft.apply_vocabulary(tokens, 
       deferred_vocab_filename_tensor=tf.constant(custom_config["token_vocab_list"]), 
       num_oov_buckets=0)

    print("print what the tokens is like")
    print(tokens)
    print(tokens.shape)

    outputs[FEATURE] = text
    outputs[_transformed_name(LABEL)] = compute_tags(labels, num_labels)
    outputs[TOKENS] = compute_tokens(tokens)

    return outputs

