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
LABEL = 'labels'

def _transformed_name(key):
    return key + '_xf'

def binarize_tags(transformed_tags, num_labels): 
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
    return tf.cast(tags_multi_binarized, tf.int64)

def preprocessing_fn(inputs, custom_config):
    """Preprocess input columns into transformed columns."""
    outputs = {}
    text = tf.squeeze(inputs[FEATURE], axis=1)
    labels = inputs[LABEL]
    tags = inputs['tags']
    
    num_labels = custom_config.get('num_labels')
    
    # Create and apply a full vocabulary for the labels (subgenres)
    labels = tft.compute_and_apply_vocabulary(
        labels, vocab_filename=LABEL, num_oov_buckets=1
    )
    # Create a full vocabulary for the tags to be accessed later
    _ = tft.vocabulary(tags, vocab_filename='tags')

    outputs[FEATURE] = text
    outputs[_transformed_name(LABEL)] = binarize_tags(labels, num_labels)

    return outputs

