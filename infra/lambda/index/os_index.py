# code to index embedding into OpenSearch
import json
import os
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import boto3

from typing import Dict, List
# check https://github.dev/hwchase17/langchain for detailed class implementation
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.llms.sagemaker_endpoint import ContentHandlerBase
import json

import logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

class ContentHandler(ContentHandlerBase):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, inputs: list, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"inputs": inputs, **model_kwargs})
        return input_str.encode('utf-8')

    def transform_output(self, output: bytes) -> List[List[float]]:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["vectors"]

content_handler = ContentHandler()

# make sure create endpoint first following instructions in embedding-model.ipynb
embeddings = SagemakerEndpointEmbeddings(
    endpoint_name="huggingface-pytorch-inference-2023-05-12-04-09-08-395", 
    region_name="us-west-2", 
    content_handler=content_handler
)

# generate embedding to allow user input simple query or documents
def generate_embedding(documents: List[str]) -> List[float]:
    return embeddings.embed_documents(documents)

# connect to OpenSearch and index the embedding
def index_embedding(embedding: List[float]):
    # Get the OpenSearch endpoint from the environment variables
    host = os.environ['OPENSEARCH_HOST']
    port = os.environ['OPENSEARCH_PORT']
    index = os.environ['OPENSEARCH_INDEX']

    # Get the AWS credentials from the environment variables
    region = os.environ['AWS_REGION']
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, region)

    # Connect to OpenSearch
    client = OpenSearch(
        hosts = [f'{host}:{port}'],
        http_auth = auth,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection
    )

    # Check the connection
    logger.info(client.info())

    # Index the embedding
    client.index(index=index, body={'embedding': embedding})

def handler(event, _context):
    logger.info("Received event: %s", event)

    # test usage, connect to open search first
    index_embedding([1,2,3])

    # Generate text embedding from user request
    try:
        embedding = generate_embedding(event['text'])
    except Exception as e:
        logger.error("Error generating embedding: %s", e)
        return {
            'statusCode': 500,
            'body': json.dumps('Error generating embedding')
        }

    # Index the embedding into OpenSearch
    try:
        index_embedding(embedding)
    except Exception as e:
        logger.error("Error indexing embedding: %s", e)
        return {
            'statusCode': 500,
            'body': json.dumps('Error indexing embedding')
        }

    return {
        'statusCode': 200,
        'body': json.dumps('Embedding indexed successfully')
    }
