import tensorflow as tf
import tensorflow_transform as tft
import os

from tfx.dsl.component.experimental.annotations import OutputDict
from tfx.dsl.component.experimental.annotations import InputArtifact
from tfx.dsl.component.experimental.annotations import OutputArtifact
from tfx.dsl.component.experimental.annotations import Parameter
from tfx.dsl.component.experimental.decorators import component

from tfx.types.standard_artifacts import Examples
from tfx.types import standard_artifacts

from tfx.utils import io_utils
from tfx.utils import path_utils

from tensorflow.keras.models import load_model
import main.components.component_utils as component_utils


@component
def ThresholdValidator(
    trained_model: InputArtifact[standard_artifacts.Model],
    transform_graph: InputArtifact[standard_artifacts.TransformGraph],
    transform_examples: InputArtifact[standard_artifacts.Examples],
    blessing: OutputArtifact[standard_artifacts.ModelBlessing],
    recall_threshold: Parameter[float] = 0.4,
    precision_threshold: Parameter[float] = 0.7,
) -> OutputDict(precision=float, recall=float):
    """Simple custom model validation component."""

    loaded_model = load_model(os.path.join(trained_model.uri, "serving_model_dir"))

    tf_transform_output = tft.TFTransformOutput(transform_graph.uri)
    num_tags = tf_transform_output.vocabulary_size_by_name("tags")
    tag_file = tf_transform_output.vocabulary_file_by_name("tags")

    eval_files = os.path.join(transform_examples.uri, "eval/*")
    table = component_utils.create_tag_lookup_table(tag_file)

    eval_dataset = component_utils._input_fn(
        file_pattern=eval_files,
        tf_transform_output=tf_transform_output,
        batch_size=64,
        epochs=1,
        num_tags=num_tags,
        table=table,
        shuffle=False,
    )
    results = loaded_model.evaluate(eval_dataset)

    if (results[1] >= precision_threshold) and (results[2] >= recall_threshold):
        io_utils.write_string_file(os.path.join(blessing.uri, "BLESSED"), "")
        blessing.set_int_custom_property("blessed", 1)
    else:
        io_utils.write_string_file(os.path.join(blessing.uri, "NOT_BLESSED"), "")
        blessing.set_int_custom_property("blessed", 0)

    return {"precision": results[1], "recall": results[2]}
