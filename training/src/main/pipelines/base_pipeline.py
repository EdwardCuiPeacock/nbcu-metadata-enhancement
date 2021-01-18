"""
TFX pipeline definition
"""

from typing import Optional, Text, List, Dict, Any
import tensorflow_model_analysis as tfma
import tensorflow as tf
from tensorflow_model_analysis.metrics import metric_specs

from ml_metadata.proto import metadata_store_pb2
from tfx.extensions.google_cloud_big_query.example_gen.component import (
    BigQueryExampleGen,
)  # pylint: disable=unused-import
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import ResolverNode
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components.trainer import executor as trainer_executor
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.dsl.component.experimental import container_component
from tfx.dsl.component.experimental import placeholders
from tfx.extensions.google_cloud_ai_platform.pusher import (
    executor as ai_platform_pusher_executor,
)
from tfx.extensions.google_cloud_ai_platform.trainer import (
    executor as ai_platform_trainer_executor,
)
from tfx.orchestration import pipeline
from tfx.proto import pusher_pb2, trainer_pb2, example_gen_pb2
from tfx.types import Channel
from tfx.types import standard_artifacts
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.utils.dsl_utils import external_input
from tfx.components import ImportExampleGen
from tfx.components.base import executor_spec
from tfx.components.example_gen.component import FileBasedExampleGen
from tfx.components.example_gen.custom_executors import parquet_executor
from tfx.components.example_gen.csv_example_gen import executor
from tfx.components import CsvExampleGen
from tfx.orchestration import data_types
from pipeline import configs


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    data_path: Text,
    query: Text,
    preprocessing_fn: Text,
    run_fn: Text,
    train_args: trainer_pb2.TrainArgs,
    eval_args: trainer_pb2.EvalArgs,
    eval_accuracy_threshold: float,
    serving_model_dir: Text,
    metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[Text]] = None,
    ai_platform_training_args: Optional[Dict[Text, Text]] = None,
    ai_platform_serving_args: Optional[Dict[Text, Any]] = None,
) -> pipeline.Pipeline:

    components = []

    # Data loading/pre-preprocessing
    data_loading_component = container_component.create_container_component(
        name="metadata-preprocessing",
        parameters={},
        # The component code uses gsutil to upload the data to GCS, so the
        # container image needs to have gsutil installed and configured.
        # Fixing b/150670779 by merging cl/294536017 will lift this limitation.
        image=configs.GCP_SDK_IMAGE_URI,
        command=["sh", "-exc", configs.PREPROCESSING_SCRIPT],
    )
    loader = data_loading_component()
    components.append(loader)

    # Parquet examples
    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=9),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=1),
            ]
        )
    )

    example_gen = FileBasedExampleGen(
        input_base=configs.DATA_PATH,
        custom_executor_spec=executor_spec.ExecutorClassSpec(parquet_executor.Executor),
        instance_name="ParquetExampleGen",
        output_config=output,
    )
    example_gen.add_upstream_node(loader)
    components.append(example_gen)

    # Computes statistics over data for visualization and example validation.
    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])
    components.append(statistics_gen)

    # Generates schema based on statistics files.
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"], infer_feature_shape=True
    )
    components.append(schema_gen)

    # Performs anomaly detection based on statistics and data schema.
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )
    components.append(example_validator)

    # Performs transformations and feature engineering in training and serving.
    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        preprocessing_fn=preprocessing_fn,
    )
    components.append(transform)

    # Uses user-provided Python function that implements a model using TF-Learn.
    trainer_args = {
        "run_fn": run_fn,
        "transformed_examples": transform.outputs["transformed_examples"],
        "schema": schema_gen.outputs["schema"],
        "transform_graph": transform.outputs["transform_graph"],
        "train_args": train_args,
        "eval_args": eval_args,
        "custom_executor_spec": executor_spec.ExecutorClassSpec(
            trainer_executor.GenericExecutor
        ),
    }
    if ai_platform_training_args is not None:
        trainer_args.update(
            {
                "custom_executor_spec": executor_spec.ExecutorClassSpec(
                    ai_platform_trainer_executor.GenericExecutor
                ),
                "custom_config": {
                    ai_platform_trainer_executor.TRAINING_ARGS_KEY: ai_platform_training_args,
                },
            }
        )
    trainer = Trainer(**trainer_args)
    components.append(trainer)

    # Get the latest blessed model for model validation.
    model_resolver = ResolverNode(
        instance_name="latest_blessed_model_resolver",
        resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    )

    components.append(model_resolver)

    # Uses TFMA to compute a evaluation statistics over features of a model and
    # perform quality validation of a candidate model (compared to a baseline).

    accuracy_threshold = tfma.MetricThreshold(
        value_threshold=tfma.GenericValueThreshold(
            lower_bound={"value": 0.0001}, upper_bound={"value": 0.99}
        ),
        change_threshold=tfma.GenericChangeThreshold(
            absolute={"value": 0.001}, direction=tfma.MetricDirection.HIGHER_IS_BETTER
        ),
    )

    metrics_specs = tfma.MetricsSpec(
        metrics=[
            tfma.MetricConfig(
                class_name="BinaryCrossentropy", threshold=accuracy_threshold
            ),
            tfma.MetricConfig(class_name="ExampleCount"),
            tfma.MetricConfig(class_name="F1Score"),
            tfma.MetricConfig(class_name="CategoricalAccuracy"),
            tfma.MetricConfig(class_name="MultiLabelConfusionMatrixPlot"),
        ]
    )

    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key="label")],
        metrics_specs=[metrics_specs],
        slicing_specs=[tfma.SlicingSpec()],
    )

    evaluator = Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        eval_config=eval_config,
    )
    components.append(evaluator)
    """
  # Evaluation how well the model predicts recommendations
  rec_eval_component = container_component.create_container_component(
    name='metadata-rec-eval',
    parameters={
    },
    image=configs.GCP_SDK_IMAGE_URI,
    command=[
        'sh', '-exc', configs.REC_EVAL_SCRIPT
    ],
  )
  rec_eval = rec_eval_component()
  rec_eval.add_upstream_node(evaluator)
  components.append(rec_eval)
  """
    # Checks whether the model passed the validation steps and pushes the model
    # to a file destination if check passed.
    pusher_args = {
        "model": trainer.outputs["model"],
        "model_blessing": evaluator.outputs["blessing"],
        "push_destination": pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    }
    if ai_platform_serving_args is not None:
        pusher_args.update(
            {
                "custom_executor_spec": executor_spec.ExecutorClassSpec(
                    ai_platform_pusher_executor.Executor
                ),
                "custom_config": {
                    ai_platform_pusher_executor.SERVING_ARGS_KEY: ai_platform_serving_args
                },
            }
        )
    # pusher = Pusher(**pusher_args)
    # components.append(pusher)

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=False,
        metadata_connection_config=metadata_connection_config,
        beam_pipeline_args=beam_pipeline_args,
    )
