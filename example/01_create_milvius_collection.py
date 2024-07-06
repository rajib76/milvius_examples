# Author: Rajib Deb
# Description: This module creates a collection in Milvius
import os

from dotenv import load_dotenv
from pymilvus import MilvusClient

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
CLUSTER_ENDPOINT= os.environ.get('CLUSTER_ENDPOINT')
TOKEN= os.environ.get('MILVIUS_TOKEN')

client = MilvusClient(
    uri=CLUSTER_ENDPOINT,
    token=TOKEN
)

client.create_collection(
    collection_name="milvius_embedding",
    dimension=1536,
    metric_type = "COSINE"
)
