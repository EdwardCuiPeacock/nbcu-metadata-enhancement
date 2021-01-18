import sys
import pandas as pd
from io import BytesIO
from tensorflow.python.lib.io import file_io

# TODO: what is text-embeddings and where does it come from?
sys.path.append("../../text-embeddings")
# TODO: improve star imports
from module.utils import *
from module.dataloader import *
from module.tokenizer import *
from module.model import *
from module.prediction_plots import *
from google.cloud import bigquery

from components.tokenizer import BertTokenizer

# load data
# TODO: where does MerlinDataLoaderWithKidsBucketsClean come from?
# data_loader = MerlinDataLoaderWithKidsBucketsClean(use_cache=False)
data_loader = {}
data_loader.prepare_data()

finaldf = pd.DataFrame()
finaldf["label"] = data_loader.data.iloc[:, 3:].values.tolist()
# tokenize data
frac, seed = 1, 11
max_seq_len = 50
data_loader.create_train_test(frac, seed)
tokenizer = BertTokenizer(max_seq_len)
padded_train = tokenizer.tokenize(data_loader.train_texts, fit=True).tolist()

# collapse genres into columns
data_loader.data = data_loader.data.astype(object)
finaldf["tokens"] = padded_train
cols = list(finaldf.columns)
cols[0], cols[1] = cols[1], cols[0]
finaldf = finaldf[cols]
finaldf.to_parquet(sys.argv[1] + ".parquet")

# append content_id column

# get id-title mappings
client = bigquery.Client()

query = """
SELECT content_ordinal_id,
       program_title
FROM recsystem.ContentOrdinalId
"""

query_job = client.query(query)
id_df = query_job.result().to_dataframe()
id_df["program_title"] = id_df["program_title"].astype(object)
id_df["content_ordinal_id"] = id_df["content_ordinal_id"].astype(object)
print(len(id_df["content_ordinal_id"].unique()))
# inner join on matching titles
finaldf["program_title"] = data_loader.data.iloc[:, 1].values.tolist()
finaldf["program_title"] = finaldf["program_title"].astype(object)
merged = pd.merge(finaldf, id_df, how="inner", on="program_title")
merged = merged.drop_duplicates(subset=["program_title"])
merged.to_parquet(sys.argv[1] + "_with_ids.parquet")
