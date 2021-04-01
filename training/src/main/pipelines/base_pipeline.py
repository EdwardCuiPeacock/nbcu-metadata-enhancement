"""
TFX pipeline definition

This is the full pipeline using the BQ example gen component
"""
import os
from typing import Optional, Text, List, Dict, Any
import tensorflow as tf

from ml_metadata.proto import metadata_store_pb2
from tfx.extensions.google_cloud_big_query.example_gen.component import (
    BigQueryExampleGen,
)  # pylint: disable=unused-import
from tfx.components import (
    ExampleValidator,
    Pusher,
    StatisticsGen,
    Trainer,
    ImporterNode,
    Transform)
from tfx.components.trainer import executor as trainer_executor
from tfx.extensions.google_cloud_ai_platform.trainer import (
    executor as ai_platform_trainer_executor,
)
from tfx.orchestration import pipeline
from tfx.proto import pusher_pb2, trainer_pb2, example_gen_pb2
from tfx.types import Channel
from tfx.types import standard_artifacts
from tfx.types.standard_artifacts import Model, ModelBlessing, Schema
from tfx.utils.dsl_utils import external_input
from tfx.dsl.components.base import executor_spec

from main.pipelines import configs


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    query: Text,
    preprocessing_fn: Text,
    run_fn: Text,
    train_args: trainer_pb2.TrainArgs,
    eval_args: trainer_pb2.EvalArgs,
    eval_accuracy_threshold: float,
    serving_model_dir: Text,
    custom_config: Dict[Text, int],
    metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[Text]] = None,
    ai_platform_training_args: Optional[Dict[Text, Text]] = None
) -> pipeline.Pipeline:

    components = []

    # Dont split the data for now. Not really sure how to set the hash buckets 
    # when there's only one split? 
    output = example_gen_pb2.Output(
             split_config=example_gen_pb2.SplitConfig(splits=[
                 example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=10)
             ],
    ))
    # Input data is in BQ
    example_gen = BigQueryExampleGen(query=query, 
                                     output_config=output)
    components.append(example_gen)

    ### Import Curated Schema ###
    # Import user-provided schema.
    schema_importer = ImporterNode(
        instance_name='import_user_schema',
        source_uri='schema/',
        artifact_type=Schema)
    components.append(schema_importer)

    # Computes statistics over data for visualization and example validation.
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs['examples'],
        schema=schema_importer.outputs['result'],
    )
    components.append(statistics_gen)

    # Performs anomaly detection based on statistics and data schema.
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_importer.outputs['result']
    )
    components.append(example_validator)

    # Performs transformations and feature engineering in training and serving.
    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_importer.outputs['result'],
        preprocessing_fn=preprocessing_fn,
        custom_config=custom_config
    )
    components.append(transform)

    # Uses user-provided Python function that implements a model using TF-Learn.
    trainer_args = {
        "run_fn": run_fn,
        "transformed_examples": transform.outputs["transformed_examples"],
        "schema": schema_importer.outputs['result'],
        "transform_graph": transform.outputs["transform_graph"],
        "train_args": train_args,
        "eval_args": eval_args,
        "custom_config": custom_config,
        "custom_executor_spec": executor_spec.ExecutorClassSpec(
            trainer_executor.GenericExecutor
        ),
    }
    if ai_platform_training_args is not None:
        trainer_args.update(
            {
                "custom_executor_spec": executor_spec.ExecutorClassSpec(
                    ai_platform_trainer_executor.GenericExecutor
                )
            }
        )
        trainer_args['custom_config'].update(
            {
                ai_platform_trainer_executor.TRAINING_ARGS_KEY: ai_platform_training_args
            }
        )
    trainer = Trainer(**trainer_args)
    components.append(trainer)

    ### PUSHER ###
    pusher = Pusher(
        model=trainer.outputs["model"],
        # model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )
    components.append(pusher)

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=False,
        metadata_connection_config=metadata_connection_config,
        beam_pipeline_args=beam_pipeline_args,
    )
