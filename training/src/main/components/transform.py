"""
For now this component mostly just passes through the data. 
One bit of processing we currently do is to join together all 
of the labels. Once (if) we move this into the query at the front 
of the pipeline, we might be able to get rid of this component 
entirely 
"""
import tensorflow as tf 

FEATURE = 'program_longsynopsis'
# TODO: Need to get the real list
LABEL_COLS = ['Action & Adventure', 'Animals', 'Animated', 'Anime', 'Art', 'Auto',
       'Biography', 'Business & Finance', 'Children\'s/Family Entertainment',
       'Comedy', 'Crime', 'Documentary', 'Drama', 'Educational',
       'Entertainment', 'Fantasy', 'Fashion', 'Fishing', 'Food', 'Game Show',
       'Health', 'History', 'Home & Garden', 'Home improvement', 'How-To',
       'Interview', 'Legal', 'Medical', 'Music', 'Mystery', 'Nature', 'News',
       'Outdoors', 'Politics & Government', 'Public Affairs', 'Reality',
       'Religion', 'Romance', 'Science & Technology', 'Science fiction',
       'Shopping', 'Sitcom', 'Soap Opera', 'Sports', 'Talk', 'Thriller',
       'Travel', 'kids (ages 5-9)', 'not for kids', 'older teens (ages 15+)',
       'preschoolers (ages 2-4)', 'teens (ages 13-14)', 'tweens (ages 10-12)']

def _transformed_name(key):
    return key + '_xf'

def preprocessing_fn(inputs):
    """Preprocess input columns into transformed columns."""
    outputs = {}
    text = tf.squeeze(inputs[FEATURE], axis=1)
    tags = tf.concat([inputs[col] for col in LABEL_COLS], 1)
    
    outputs[_transformed_name(FEATURE)] = text
    outputs['tags'] = tags
    
    return outputs