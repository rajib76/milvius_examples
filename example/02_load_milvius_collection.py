# Author: Rajib Deb
# Description: This module loads embedding
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pymilvus import MilvusClient, DataType

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
CLUSTER_ENDPOINT = os.environ.get('CLUSTER_ENDPOINT')
TOKEN = os.environ.get('MILVIUS_TOKEN')

client = MilvusClient(
    uri=CLUSTER_ENDPOINT,
    token=TOKEN
)

MODEL_NAME = "text-embedding-3-small"  # Which model to use, please check https://platform.openai.com/docs/guides/embeddings for available models
DIMENSION = 1536  # Dimension of vector embedding

# Connect to OpenAI with API Key.
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Extracting content from the document
with open("/Users/joyeed/milvius/milvius_examples/data/obama_speech.txt") as f:
    obama_speech_content = f.read()

# Splitting into chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=100,
    chunk_overlap=0,
)
texts = text_splitter.split_text(obama_speech_content)
documents = []
for text in texts:
    document = Document(page_content=text, metadata={"source": "obama_speech.txt"})
    documents.append(document)

docs = []
sources = []
for doc in documents:
    docs.append(doc.page_content)
    sources.append(doc.metadata['source'])


vectors = [
    vec.embedding
    for vec in openai_client.embeddings.create(input=docs, model=MODEL_NAME).data
]

data = [
    {"id": i, "vector": vectors[i], "text": docs[i], "source": sources[i]}
    for i in range(len(docs))
]

res = client.insert(collection_name="milvius_embedding", data=data)
