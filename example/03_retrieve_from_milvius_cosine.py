# Author: Rajib Deb
# Description: This module searches similar chunk based on cosine similarity
import os

from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import MilvusClient

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
CLUSTER_ENDPOINT= os.environ.get('CLUSTER_ENDPOINT')
TOKEN= os.environ.get('MILVIUS_TOKEN')

client = MilvusClient(
    uri=CLUSTER_ENDPOINT,
    token=TOKEN
)
MODEL_NAME = "text-embedding-3-small"
DIMENSION = 1536

queries = ["How many Americans develop Alzheimer's in a year"]

# Connect to OpenAI with API Key.
openai_client = OpenAI(api_key=OPENAI_API_KEY)

query_vectors = [
    vec.embedding
    for vec in openai_client.embeddings.create(input=queries, model=MODEL_NAME).data
]

# Refer the below link to see the configuration settings
# https://milvus.io/docs/v2.3.x/within_range.md
param = {
    # use `COSINE` as the metric to calculate the similarity
    "metric_type": "COSINE",
    "params": {
        # search for vectors with score greater than 0.5
        "radius": 0.5,
        # filter out most similar vectors with a score less than or equal to 1.0
        "range_filter" : 1.0
    }
}
res = client.search(
    collection_name="milvius_embedding",  # target collection
    data=query_vectors,  # query vectors
    limit=3,  # number of returned entities
    output_fields=["text", "source"],  # specifies fields to be returned,
    search_params=param
)

for q in queries:
    print("Query:", q)
    for result in res:
        print(result)
    print("\n")