# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluator Component for Metadata TFX custom component.

This custom component simply passes examples through. This is meant to serve as
a kind of starting point example for creating custom components.

This component along with other custom component related code will only serve as
an example and will not be supported by TFX team.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from typing import Any, Dict, List, Text

import tensorflow_text
import tensorflow as tf
from tfx import types
from tfx.dsl.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import io_utils, path_utils

from google.cloud import bigquery
import pandas as pd
import numpy as np
from scipy import spatial
import sys
import random
import datetime

import subprocess
import sys

TITLES_QUERY = """
SELECT 
    DISTINCT
    TitleDetails_title, 
    TitleType, 
    STRING_AGG(DISTINCT TitleDetails_longsynopsis, ' ') as TitleDetails_longsynopsis, 
    cid.content_ordinal_id,
FROM `res-nbcupea-dev-ds-sandbox-001.metadata_enhancement.ContentMetadataView` cmv
LEFT JOIN `res-nbcupea-dev-ds-sandbox-001.recsystem.ContentOrdinalId` cid
    ON cmv.TitleDetails_title = cid.program_title
WHERE 
    TitleDetails_longsynopsis IS NOT NULL
    AND cid.content_ordinal_id IS NOT NULL
GROUP BY 
    TitleDetails_title, 
    TitleType, 
    cid.content_ordinal_id
"""

date_start = "2021-2-01"
date_end = "2021-4-01"

PREV_WINDOW = 20
TEST_WINDOW = 5
DATA_LENGTH = PREV_WINDOW + TEST_WINDOW
NUM_SAMPLES = 50000

USERS_QUERY = ("""
SELECT psv2.user_ordinal_id, 
       session_date, 
       session_timestamp, 
       ss.content_id
FROM `res-nbcupea-dev-ds-sandbox-001.recsystem.PlaySequenceV2` as psv2,
UNNEST(session_sequence) AS ss
LEFT JOIN (
SELECT user_ordinal_id, 
       COUNT(DISTINCT(ss.content_id)),
       row_number() over (order by farm_fingerprint(concat(user_ordinal_id, '3')) ) as seqnum
FROM `res-nbcupea-dev-ds-sandbox-001.recsystem.PlaySequenceV2`, 
UNNEST(session_sequence) AS ss
WHERE session_date >= "{0}"
      AND session_date < "{1}"
GROUP BY user_ordinal_id
HAVING COUNT(DISTINCT(ss.content_id)) > {2}
) filtered_data
ON filtered_data.user_ordinal_id = psv2.user_ordinal_id
WHERE seqnum <= {3}
ORDER BY session_timestamp
""").format(date_start, date_end, DATA_LENGTH, NUM_SAMPLES)


def ntail(g, n):
    return g._selected_obj[g.cumcount() >= n]

class Executor(base_executor.BaseExecutor):
    """Executor for Evaluator Component for Metadata."""

    def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
        """Copy the input_data to the output_data.

        For this example that is all that the Executor does.  For a different
        custom component, this is where the real functionality of the component
        would be included.

        This component both reads and writes Examples, but a different component
        might read and write artifacts of other types.

        Args:
          input_dict: Input dict from input key to a list of artifacts, including:
            - input_data: A list of type `standard_artifacts.Examples` which will
              often contain two splits, 'train' and 'eval'.
          output_dict: Output dict from key to a list of artifacts, including:
            - output_data: A list of type `standard_artifacts.Examples` which will
              usually contain the same splits as input_data.
          exec_properties: A dict of execution properties, including:
            - name: Optional unique name. Necessary iff multiple Hello components
              are declared in the same pipeline.

        Returns:
          None

        Raises:
          OSError and its subclasses
        """

        ###Package installation
        #install(exec_properties['packages'])

        ### Load embeddings
        """
        embeddings = artifact_utils.get_single_instance(
            input_dict['embeddings'])
        embeddings_path = path_utils.serving_model_path(embeddings.uri)
        """
        
        model = artifact_utils.get_single_instance(
            input_dict['model'])
        model_path = path_utils.serving_model_path(model.uri)

        model = tf.keras.models.load_model(model_path)
        
        ### Load User Data
        """
        user_data = artifact_utils.get_single_instance(
            input_dict['user_data'])
        user_data_path = path_utils.serving_model_path(user_data.uri)
        
        ### Run evaluation, return metrics (dict of metric name to number)
        metrics = exec_properties['run_fn'](embeddings,
                                            user_data)
        """
        
        client = bigquery.Client()
        raw_user_data = client.query(USERS_QUERY).result().to_dataframe()
        
        ### Create embeddings
        unscored_titles = client.query(TITLES_QUERY) \
                                .result() \
                                .to_dataframe() \
                                .drop_duplicates(subset=['TitleDetails_title']) \
                                .reset_index()

        input_data = unscored_titles[['TitleDetails_longsynopsis']]
        dataset = tf.data.Dataset.from_tensor_slices(
                {'synopsis': tf.cast(input_data['TitleDetails_longsynopsis'].values.tolist(), tf.string)}
                ).batch(50)
        
        res = []
        for element in dataset:
            y = model.predict(element)
            res.append(y)
        
        f = tf.concat(res, axis=0).numpy()
        preds = pd.DataFrame(f)
        preds['pred'] = preds.iloc[:,:].values.tolist()
        preds['pred'] = preds['pred'].apply(np.asarray)
        preds['content_ordinal_id'] = unscored_titles['content_ordinal_id']
        preds = preds[['pred','content_ordinal_id']]
        
        user_data = pd.merge(raw_user_data, 
                             preds,
                             how="inner", 
                             left_on="content_id", 
                             right_on="content_ordinal_id") \
                             .drop(columns=['content_ordinal_id'])
        
        test_window_data = user_data.sort_values(['user_ordinal_id','session_timestamp']) \
                                        .groupby(['user_ordinal_id'],sort=True)
        
        history_df = test_window_data.head(PREV_WINDOW).reset_index()
        
        # get all entries after the first 20
        after_prev = ntail(test_window_data, PREV_WINDOW).reset_index().drop_duplicates(['user_ordinal_id', 'content_id'])
        
        #remove content that was in the history_df
        after_prev = pd.merge(after_prev,
                              history_df[['user_ordinal_id', 'content_id', 'index']],
                              how="left",
                              on=['user_ordinal_id', 'content_id'])
        
        # remove entries where the show was already watched in history
        after_prev = after_prev.loc[after_prev['index_y'].isnull()] 
        
        test_df = after_prev.sort_values(['user_ordinal_id', 'session_timestamp']) \
                            .groupby(['user_ordinal_id'],sort=True) \
                            .head(TEST_WINDOW) \
                            .drop(columns=['index_y','index_x']) \
                            .reset_index()

        avg_emb = history_df.groupby(['user_ordinal_id'])['pred'] \
                            .apply(np.sum)

        avg_emb = avg_emb.apply(lambda x: np.asarray([i / PREV_WINDOW for i in x]))

        cos_sim = avg_emb.apply(lambda x: [1 - spatial.distance.cosine(u, x) for u in preds['pred']])
        
        # make cosine sim for things in the history -1 so they don't get predicted
        for x in cos_sim.index:
            user_hist = history_df.loc[history_df['user_ordinal_id'] == x].content_id.unique().tolist()
            histidx = preds.loc[preds['content_ordinal_id'].isin(user_hist)]
            for i in histidx.index:
                cos_sim[x][i] = -1

        ## Predict / Eval on test-data using cos sim
        # prev_window: how many prior shows to average into user embedding
        # test_window: how many shows to allow into the future for correctly guessing.
        #              i.e test_window=1 means predict next show exactly,
        #                  test_window=5 means prediction must be within next 5 shows watched

        counter = 0

        correct = 0

        recall = {}
        precision = {}
        coverage = {}
        seen = {}
        accuracy = {}
        total = len(user_data['user_ordinal_id'].unique())

        for n in [-1,-5,-10]:

            top = cos_sim.apply(lambda x: np.argsort(x)[n:])
            #topn = top.apply(lambda x: x[n:])
            top_with_ids = top.apply(lambda x: set([preds['content_ordinal_id'][i] for i in x])).reset_index()

            for _ , userid in top_with_ids.iterrows():

                future_data = set(test_df.loc[test_df['user_ordinal_id'] == userid['user_ordinal_id']].content_id.values.tolist())
                topn = top_with_ids.loc[top_with_ids['user_ordinal_id'] == userid['user_ordinal_id']].pred.values[0]
                
                if n in coverage.keys():
                    coverage[n] = coverage[n].union(topn)
                    seen[n] = seen[n].union(future_data)
                    recall[n].append(len(topn.intersection(future_data)) / TEST_WINDOW)
                    precision[n].append(len(topn.intersection(future_data)) / -n)
                else:
                    coverage[n] = topn
                    seen[n] = future_data
                    recall[n] = [len(topn.intersection(future_data))]
                    precision[n] = [len(topn.intersection(future_data)) / -n]
                if len(topn.intersection(future_data)) >= 1:

                    if n not in accuracy.keys():
                        accuracy[n] = 1
                    else:
                        accuracy[n] += 1

        metric_vals = {}
        metric_vals['Precision@1'] = sum(precision[-1]) / total
        metric_vals['Precision@5'] = sum(precision[-5]) / total
        metric_vals['Precision@10'] = sum(precision[-10]) / total

        metric_vals['Recall@1'] = sum(recall[-1]) / total
        metric_vals['Recall@5'] = sum(recall[-5]) / total
        metric_vals['Recall@10'] = sum(recall[-10]) / total

        metric_vals['Accuracy@1'] = accuracy[-1] / total
        metric_vals['Accuracy@5'] = accuracy[-5] / total
        metric_vals['Accuracy@10'] = accuracy[-10] / total

        metric_vals['Coverage@1'] = len(coverage[-1]) / len(seen[-1])
        metric_vals['Coverage@5'] = len(coverage[-5]) / len(seen[-5])
        metric_vals['Coverage@10'] = len(coverage[-10]) / len(seen[-10])
        ### Write Metrics
        client = bigquery.Client()
        date = datetime.datetime.now().strftime("%Y-%2m-%d %H:%M:%S")

        for k,v in metric_vals.items():
            row_to_insert = {}
            row_to_insert['model_name'] = exec_properties['name']
            row_to_insert['date'] = date
            row_to_insert['metric_name'] = k
            row_to_insert['value'] = v
            row_to_insert['model_path'] = model_path
            
            errors = client.insert_rows_json(exec_properties['output_table'], [row_to_insert])
        print("errors: ", errors)

        
        