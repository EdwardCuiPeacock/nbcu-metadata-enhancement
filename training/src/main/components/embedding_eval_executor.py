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
import gcsfs
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
import time

import subprocess
import sys

from main.pipelines import configs

TITLES_QUERY_old = """
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

TITLES_QUERY_tokens = """
    WITH titles_data AS (SELECT 
        TitleDetails_title, 
        TitleType,
        STRING_AGG(DISTINCT TitleDetails_longsynopsis, ' ') AS TitleDetails_longsynopsis,
        SPLIT(STRING_AGG(DISTINCT TitleDetails_longsynopsis, ' '), " ") AS synopsis_list, 
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
        cid.content_ordinal_id),
    
    raw_tags AS (
        SELECT TitleDetails_title, ss AS tokens
        FROM titles_data,
        UNNEST(synopsis_list) ss WITH OFFSET index
        WHERE index BETWEEN 1 AND 256
    ),
        
    tags_data AS (
        SELECT a.TitleDetails_title, ARRAY_AGG(DISTINCT a.tokens) AS tokens
        FROM raw_tags a
        INNER JOIN `res-nbcupea-dev-ds-sandbox-001.metadata_enhancement.node2vec_token_edc_dev` tk
        ON tk.tokens=a.tokens
        GROUP BY a.TitleDetails_title
    ),
    
    unfiltered AS (SELECT a.TitleDetails_title, a.TitleType, a.TitleDetails_longsynopsis,
        a.content_ordinal_id, 
        CASE
            WHEN ARRAY_LENGTH(b.tokens) < 1 OR ARRAY_LENGTH(b.tokens) IS NULL THEN ["TV"]
            ELSE b.tokens
        END AS tokens,
    FROM titles_data a
    LEFT JOIN tags_data b
    ON a.TitleDetails_title=b.TitleDetails_title)
    
    SELECT TitleDetails_title, TitleType, TitleDetails_longsynopsis, content_ordinal_id, 
        ARRAY_AGG(DISTINCT tk) AS tokens
    FROM unfiltered,
    UNNEST(tokens) tk WITH OFFSET index
    WHERE index BETWEEN 1 AND 64
    GROUP BY TitleDetails_title, TitleType, TitleDetails_longsynopsis, content_ordinal_id
"""

TITLES_QUERY_keywords = """
    CREATE TEMP FUNCTION strip_str_array(val ANY TYPE) AS ((
      SELECT ARRAY_AGG(DISTINCT TRIM(t))
      FROM UNNEST(val) t
      WHERE t != ""
    ));
    
    WITH titles_data AS (
        SELECT DISTINCT
            TitleDetails_title, 
            TitleType, 
            cid.content_ordinal_id,
            STRING_AGG(DISTINCT TitleDetails_longsynopsis, ' ') as TitleDetails_longsynopsis,
            STRING_AGG(DISTINCT TitleTags, ',') as TitleTags,
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
        )
    SELECT TitleDetails_title, TitleType, content_ordinal_id, TitleDetails_longsynopsis, 
        strip_str_array(SPLIT(CONCAT("movie", ",", TitleTags), ",")) AS tokens
    FROM titles_data
"""

TITLES_QUERY_token_keyword = """
    CREATE TEMP FUNCTION strip_str_array(val ANY TYPE) AS ((
      SELECT ARRAY_AGG(DISTINCT TRIM(t))
      FROM UNNEST(val) t
      WHERE t != ""
    ));
    
    WITH titles_data AS (SELECT 
        TitleDetails_title, 
        TitleType,
        STRING_AGG(DISTINCT TitleDetails_longsynopsis, ' ') AS TitleDetails_longsynopsis,
        STRING_AGG(DISTINCT TitleTags, ',') as TitleTags,
        SPLIT(STRING_AGG(DISTINCT TitleDetails_longsynopsis, ' '), " ") AS synopsis_list, 
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
        cid.content_ordinal_id),
    
    raw_tags AS (
        SELECT TitleDetails_title, ss AS tokens
        FROM titles_data,
        UNNEST(synopsis_list) ss WITH OFFSET index
        WHERE index BETWEEN 1 AND 256
    ),
        
    tags_data AS (
        SELECT a.TitleDetails_title, ARRAY_AGG(DISTINCT a.tokens) AS tokens
        FROM raw_tags a
        INNER JOIN `res-nbcupea-dev-ds-sandbox-001.metadata_enhancement.node2vec_token_edc_dev` tk
        ON tk.tokens=a.tokens
        GROUP BY a.TitleDetails_title
    ),
    
    unfiltered AS (SELECT a.TitleDetails_title, a.TitleType, a.TitleDetails_longsynopsis,
        a.TitleTags, a.content_ordinal_id, 
        CASE
            WHEN ARRAY_LENGTH(b.tokens) < 1 OR ARRAY_LENGTH(b.tokens) IS NULL THEN ["TV"]
            ELSE b.tokens
        END AS tokens,
    FROM titles_data a
    LEFT JOIN tags_data b
    ON a.TitleDetails_title=b.TitleDetails_title),
    
    preproc AS(SELECT TitleDetails_title, TitleType, TitleTags, TitleDetails_longsynopsis, 
        content_ordinal_id, ARRAY_AGG(DISTINCT tk) AS tokens, 
    FROM unfiltered,
    UNNEST(tokens) tk WITH OFFSET index
    WHERE index BETWEEN 1 AND 64
    GROUP BY TitleDetails_title, TitleType,TitleTags,
        TitleDetails_longsynopsis, content_ordinal_id)
    
    SELECT TitleDetails_title, TitleType, TitleDetails_longsynopsis, content_ordinal_id, tokens,
        strip_str_array(SPLIT(CONCAT("movie", ",", TitleTags), ",")) AS keywords
    FROM preproc
"""

date_start = "2021-2-01"
date_end = "2021-4-01"

PREV_WINDOW = 10
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
        print("Loading saved model")
        model = artifact_utils.get_single_instance(
            input_dict['model'])
        model_path = path_utils.serving_model_path(model.uri)

        # model = tf.keras.models.load_model(model_path)
        model = tf.saved_model.load(model_path)
        
        ### Load User Data
        """
        user_data = artifact_utils.get_single_instance(
            input_dict['user_data'])
        user_data_path = path_utils.serving_model_path(user_data.uri)
        
        ### Run evaluation, return metrics (dict of metric name to number)
        metrics = exec_properties['run_fn'](embeddings,
                                            user_data)
        """
        print("Loading data")
        client = bigquery.Client()
        raw_user_data = client.query(USERS_QUERY).result().to_dataframe()

        ### Create embeddings
        unscored_titles = client.query(TITLES_QUERY_keywords) \
                                .result() \
                                .to_dataframe() \
                                .drop_duplicates(subset=['TitleDetails_title']) \
                                .reset_index()
        print("Start making predictions on synopsis")
        tnow = time.time()
        res = []
        if True: # using tokens
            input_data = {"synopsis": unscored_titles['TitleDetails_longsynopsis'].values[:, None], 
              "tokens": tf.ragged.constant(unscored_titles["tokens"].values).to_sparse(),
             #"kewords": tf.ragged.constant(unscored_titles["keywords"].values).to_sparse(),
             }
            dataset = tf.data.Dataset.from_tensor_slices(input_data).batch(50)

            for batch in dataset:
                transformed_features = model.tft_layer(batch)
                transformed_features["synopsis"] = transformed_features["synopsis"][:, None]
                y = model(transformed_features)
                res.append(y)
        else:
            input_data = unscored_titles[['TitleDetails_longsynopsis']]
            dataset = tf.data.Dataset.from_tensor_slices(
                    tf.cast(input_data['TitleDetails_longsynopsis'].values.tolist(), tf.string)
                    ).batch(50)
            for element in dataset:
                y = model(element)
                res.append(y)

        used_time = time.time() - tnow
        print(f"Successfully made predictions on synopsis: {used_time:.2f} s")
        
        ######## TODO: Look at this block ####################
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

        tnow = time.time()
        avg_emb = avg_emb.apply(lambda x: np.asarray([i / PREV_WINDOW for i in x]))
        used_time = time.time() - tnow
        print(f"Average embed: {used_time:.2f} s")

        ##############################################################

        tnow = time.time()
        #cos_sim = avg_emb.apply(lambda x: [1 - spatial.distance.cosine(u, x) for u in preds['pred']])
        avg_emb_mat = np.stack(avg_emb.values)
        preds_emb_mat = np.stack(preds["pred"].values)
        # L2 Normalize
        avg_emb_norm = avg_emb_mat / np.sqrt(np.sum(avg_emb_mat**2, axis=1, keepdims=True))
        preds_emb_norm = preds_emb_mat / np.sqrt(np.sum(preds_emb_mat**2, axis=1, keepdims=True))
        # Cosine similarity Scoring
        user_content_mat = avg_emb_norm @ preds_emb_norm.T
        print("cosine similarity")
        print("mean:", np.nanmean(user_content_mat.ravel()))
        print("std:", np.nanstd(user_content_mat.ravel()))
        print("median:", np.nanmedian(user_content_mat.ravel()))
        user_content_mat = np.nan_to_num(user_content_mat, nan=-1)
        cos_sim = pd.Series(list(user_content_mat), index=avg_emb.index, name="user_ordinal_id")
        used_time = time.time() - tnow
        print(f"Cosine sim: {used_time:.2f}s")

        # make cosine sim for things in the history -1 so they don't get predicted
        tnow = time.time()
        df_joined_history = history_df[["user_ordinal_id", "content_id"]].merge(
            preds.reset_index(drop=False)[["index", "content_ordinal_id"]], 
            left_on="content_id", right_on="content_ordinal_id")
        df_user_hist = df_joined_history.groupby(by=["user_ordinal_id"]).agg({"index":list})
        df_user_hist = df_user_hist.join(cos_sim.to_frame(name="cos_sim"), on="user_ordinal_id")

        def zeroing_func(pdf):
            pdf["cos_sim"][pdf["index"]] = -1
            return pdf

        df_user_hist = df_user_hist.apply(zeroing_func, axis=1)
        # Reset cos_sim
        cos_sim = df_user_hist["cos_sim"]
        print(f"cos sim slicer for loop: {used_time} s")

        ## Predict / Eval on test-data using cos sim
        # prev_window: how many prior shows to average into user embedding
        # test_window: how many shows to allow into the future for correctly guessing.
        #              i.e test_window=1 means predict next show exactly,
        #                  test_window=5 means prediction must be within next 5 shows watched

        recall = {}
        precision = {}
        coverage = {}
        seen = {}
        accuracy = {}
        total = len(user_data['user_ordinal_id'].unique())

        tnow = time.time()
        top_all = cos_sim.apply(lambda x: np.argsort(x))
        for n in [-1,-5,-10]:
            print(f"top {n}")
            top_with_ids = np.take(preds["content_ordinal_id"].values, np.stack(top_all.values)[:, n:])
            top_with_ids = pd.Series(list(top_with_ids), index=cos_sim.index, name="user_ordinal_id")\
                .to_frame(name="content_ordinal_id").reset_index()
            print("Computing metrics for each user")
            for _ , userid in top_with_ids.iterrows():
                future_data = set(test_df.loc[test_df['user_ordinal_id'] == userid['user_ordinal_id']].content_id.values.tolist())
                if len(future_data) < 1:
                    continue
                topn = set(userid["content_ordinal_id"])

                if n in coverage.keys():
                    coverage[n] = coverage[n].union(topn)
                    seen[n] = seen[n].union(future_data)
                    recall[n].append(len(topn.intersection(future_data)) / len(future_data))
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
        used_time = time.time() - tnow
        print(f"Double for loop to get top 1/5/10 metrics: {used_time} s")

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

        # Print the results
        for k, v in metric_vals.items():
            print(k, ":", v)
                
        # save metrics into json
        metrics_series = {
            "total": total,
            "coverage": {k:list(map(int, v)) for k, v in coverage.items()},
            "accuracy": accuracy,
            "precision": {k:list(map(float, v)) for k, v in precision.items()},
            "recall": {k:list(map(float, v)) for k, v in recall.items()},
            "seen": {k:list(map(int, v)) for k, v in seen.items()},
            "summary": metric_vals,
        }

        fs = gcsfs.GCSFileSystem(project="res-nbcupea-dev-ds-sandbox-001")
        time_stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        pipeline_root = configs.PIPELINE_ROOT
        with fs.open(f"{pipeline_root}/Evaluator/metrics/metrics_series_{time_stamp}.json", "w") as fid:
            json.dump(metrics_series, fid)

        ### Write Metrics
        client = bigquery.Client()
        date = datetime.datetime.now().strftime("%Y-%2m-%d %H:%M:%S")

        tnow = time.time()
        for k,v in metric_vals.items():
            row_to_insert = {}
            row_to_insert['model_name'] = exec_properties['name']
            row_to_insert['date'] = date
            row_to_insert['metric_name'] = k
            row_to_insert['value'] = v
            row_to_insert['model_path'] = model_path
            
            errors = client.insert_rows_json(exec_properties['output_table'], [row_to_insert])
       
        print("errors: ", errors)
        used_time = time.time() - tnow
        print(f"For loop to insert data in bigquery: {used_time} s")

        
        