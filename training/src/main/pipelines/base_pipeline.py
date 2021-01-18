"""
Example pipeline file, edit this file with your own pipeline
"""
from typing import Optional, Text, List, Dict
import tensorflow as tf
import tensorflow_model_analysis as tfma
import tensorflow_data_validation as tfdv
import tensorflow_transform as tft

import os

import tfx
from tfx.components import (
    CsvExampleGen,
    Evaluator,
    ExampleValidator,
    Pusher,
    ImporterNode,
    StatisticsGen,
    Trainer,
    Transform,
    ImporterNode,
)

from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor
from tfx.orchestration import pipeline
from tfx.components.example_gen.custom_executors import parquet_executor
from tfx.components.example_gen.component import FileBasedExampleGen
from tfx.extensions.google_cloud_ai_platform.trainer import (
    executor as ai_platform_trainer_executor,
)

from tfx.proto import pusher_pb2, trainer_pb2, example_gen_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing, Schema
from tfx.utils.dsl_utils import external_input

from ml_metadata.proto import metadata_store_pb2

from main.components.ThresholdValidator import ThresholdValidator
from main.pipelines import config


####### Pipeline
def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    data_path: Text,
    preprocessing_fn: Text,
    run_fn: Text,
    train_args: trainer_pb2.TrainArgs,
    eval_args: trainer_pb2.EvalArgs,
    eval_precision_threshold: float,
    eval_recall_threshold: float,
    serving_model_dir: Text,
    metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[Text]] = None,
    ai_platform_training_args: Optional[Dict[Text, Text]] = None,
) -> pipeline.Pipeline:
    """
    Implement the semantic tagging pipeline with TFX
    """
    components = []

    ### EXAMPLE GEN ###
    # Use pre-split data
    input_config = example_gen_pb2.Input(
        splits=[
            example_gen_pb2.Input.Split(name="train", pattern="train*"),
            example_gen_pb2.Input.Split(name="eval", pattern="eval*"),
        ]
    )
    # Input data is in parquet files
    example_gen = FileBasedExampleGen(
        input_base=data_path,
        custom_executor_spec=executor_spec.ExecutorClassSpec(parquet_executor.Executor),
        input_config=input_config,
    )
    components.append(example_gen)

    ### Import Curated Schema ###
    # Import user-provided schema.
    schema_importer = ImporterNode(
        instance_name="import_user_schema", source_uri="schema/", artifact_type=Schema
    )
    components.append(schema_importer)

    ### STATISTICS GEN ###
    stats_options = tfdv.StatsOptions(
        num_rank_histogram_buckets=config.NUM_RANK_HISTOGRAM
    )
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs["examples"],
        schema=schema_importer.outputs["result"],
        stats_options=stats_options,
    )
    components.append(statistics_gen)

    ### EXAMPLE VALIDATOR ###
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_importer.outputs["result"],
    )

    components.append(example_validator)

    ### TRANSFORM ###
    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_importer.outputs["result"],
        preprocessing_fn=preprocessing_fn,
    )
    components.append(transform)

    ### TRAINER ###
    trainer_args = {
        "run_fn": run_fn,
        "transformed_examples": transform.outputs["transformed_examples"],
        "schema": schema_importer.outputs["result"],
        "transform_graph": transform.outputs["transform_graph"],
        "train_args": train_args,
        "eval_args": eval_args,
        "custom_executor_spec": executor_spec.ExecutorClassSpec(GenericExecutor),
    }

    if ai_platform_training_args is not None:
        trainer_args.update(
            {
                "custom_executor_spec": executor_spec.ExecutorClassSpec(
                    ai_platform_trainer_executor.GenericExecutor
                ),
                "custom_config": {
                    ai_platform_trainer_executor.TRAINING_ARGS_KEY: ai_platform_training_args
                },
            }
        )

    trainer = Trainer(**trainer_args)
    components.append(trainer)

    ### EVALUATOR ###
    evaluator = ThresholdValidator(
        trained_model=trainer.outputs["model"],
        transform_graph=transform.outputs["transform_graph"],
        transform_examples=transform.outputs["transformed_examples"],
        recall_threshold=eval_recall_threshold,
        precision_threshold=eval_precision_threshold,
    )
    components.append(evaluator)

    ### PUSHER ###
    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
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
        enable_cache=True,
        metadata_connection_config=metadata_connection_config,
        beam_pipeline_args=beam_pipeline_args,
    )
