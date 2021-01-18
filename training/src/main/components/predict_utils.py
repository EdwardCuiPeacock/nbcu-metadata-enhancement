import googleapiclient.discovery
from tokenizer import BertTokenizer
import base64
import numpy as np
import time
import tensorflow as tf


def predict_json(project, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the AI Platform Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    # Create the AI Platform service object.
    # To authenticate set the environment variable
    # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
    service = googleapiclient.discovery.build("ml", "v1")
    name = "projects/{}/models/{}".format(project, model)

    if version is not None:
        name += "/versions/{}".format(version)

    response = (
        service.projects().predict(name=name, body={"instances": instances}).execute()
    )

    if "error" in response:
        raise RuntimeError(response["error"])

    return response["predictions"]


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(tokens):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {"tokens": _int64_feature(tokens), "label": _int64_feature(np.arange(32))}
    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def serialize_synopsis(synopsis: str, max_seq_len, bert_max=None):

    tokenizer = BertTokenizer(max_seq_len, bert_max)
    tokenized = tokenizer.tokenize([synopsis])
    serialized = serialize_example(tokenized[0])
    serialized = base64.b64encode(serialized).decode("utf-8")

    return serialized


def request_synopsis_embedding(
    synopsis, max_seq_len, GOOGLE_CLOUD_PROJECT, model_name, version
):
    """

    Output
    ------
    Weights: List[List[Float]]
    Genres in the following order for movies:
    'Action & Adventure', 'Animated', 'Anime', 'Biography', "Children's/Family Entertainment", 'Comedy', 'Courtroom', 'Crime',
    'Documentary', 'Drama', 'Educational', 'Fantasy', 'Gay and Lesbian', 'History', 'Holiday', 'Horror', 'Martial arts', 'Military & War',
    'Music', 'Musical', 'Mystery', 'Romance', 'Science fiction', 'Sports', 'Thriller', 'Western', 'kids (ages 5-9)', 'not for kids', 'older teens (ages 15+)',
    'preschoolers (ages 2-4)', 'teens (ages 13-14)', 'tweens (ages 10-12)'
    """
    t = time.time()
    serialized = serialize_synopsis(synopsis, max_seq_len)
    instances = [{"examples": {"b64": serialized}}]
    print(time.time() - t)
    t = time.time()
    pred = predict_json(GOOGLE_CLOUD_PROJECT, model_name, instances, version)
    print(time.time() - t)
    return pred
