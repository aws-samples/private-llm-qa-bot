# code to query embedding from OpenSearch
import json
import os
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

def handler(event, context):
    # Query OpenSearch with the provided parameters
    results = query_opensearch(event['queryStringParameters'])

    return {
        'statusCode': 200,
        'body': json.dumps(results)
    }

def query_opensearch(query_params):
    region = os.environ['AWS_REGION']
    service = 'es'
    index_name = 'embeddings'

    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)

    opensearch = OpenSearch(
        hosts=[os.environ['OPENSEARCH_ENDPOINT']],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )

    # Replace with your OpenSearch query logic
    search_body = {
        "query": {
            "match_all": {}
        }
    }

    response = opensearch.search(index=index_name, body=search_body)

    return response['hits']['hits']