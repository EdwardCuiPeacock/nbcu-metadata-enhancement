"""
TFX Peacock Recs model preprocessing
"""

# from __future__ import division
# from __future__ import print_function

import tensorflow as tf
import tensorflow_transform as tft

# from models import features


def preprocessing_fn(inputs):
    output = {}
    #     label_columns = ['Action & Adventure', 'Animated', 'Anime', 'Biography',
    #        "Children's/Family Entertainment", 'Comedy', 'Courtroom', 'Crime',
    #        'Documentary', 'Drama', 'Educational', 'Fantasy', 'Gay and Lesbian',
    #        'History', 'Holiday', 'Horror', 'Martial arts', 'Military & War',
    #        'Music', 'Musical', 'Mystery', 'Romance', 'Science fiction', 'Sports',
    #        'Thriller', 'Western', 'kids (ages 5-9)', 'not for kids',
    #        'older teens (ages 15+)', 'preschoolers (ages 2-4)',
    #        'teens (ages 13-14)', 'tweens (ages 10-12)']

    #     label = [inputs[i] for i in label_columns]
    #     output['label'] = tf.concat(label, axis = -1)
    output["tokens"] = inputs["tokens"]
    output["label"] = inputs["label"]

    return output
