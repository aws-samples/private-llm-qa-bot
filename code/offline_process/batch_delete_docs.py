import logging
import boto3
import os
import time
import pytz
import json
import argparse
from datetime import datetime
from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection

logger = logging.getLogger()
logger.setLevel(logging.INFO)

awsauth=None
region=None
DOC_INDEX_TABLE= 'chatbot_doc_index'


def list_s3_objects(s3_client,bucket_name, prefix=''):
    objects = []
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    # iterate over pages
    for page in page_iterator:
        # loop through objects in page
        if 'Contents' in page:
            for obj in page['Contents']:
                yield obj['Key']
        # if there are more pages to fetch, continue
        if 'NextContinuationToken' in page:
            page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix,
                                                ContinuationToken=page['NextContinuationToken'])

def delete_doc_index(aos_endpoint, obj_key, embedding_model, index_name):
    global region
    def delete_aos_index(aos_endpoint, obj_key, index_name, size=50):
        global awsauth
        client = OpenSearch(
                    hosts=[{'host':aos_endpoint, 'port': 443}],
                    http_auth = awsauth,
                    use_ssl=True,
                    verify_certs=True,
                    connection_class=RequestsHttpConnection
                )
        
        data = {
            "size": size,
            "query" : {
                "match_phrase":{
                    "doc_title": obj_key
                }
            }
        }

        response = client.delete_by_query(index=index_name, body=data)
        
        try:
            if 'deleted' in response.keys():
                logger.info(f"delete:{obj_key}, {response['deleted']} records was deleted")
                return True
            else:
                logger.info(f"delete:{obj_key} failed.")
        except Exception as e:
            logger.info(f"delete:{obj_key} failed, caused by {str(e)}")


        return False

    success = delete_aos_index(aos_endpoint, obj_key, index_name)

    if success:
        ##删除ddb里的索引
        dynamodb = boto3.client('dynamodb', region)
        try:
            dynamodb.delete_item(
                TableName=DOC_INDEX_TABLE,
                Key={
                    'filename': {'S': obj_key},
                    'embedding_model': {'S': embedding_model}
                }
            )
        except Exception as e:
            logger.info(str(e))
            return False

    return success

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', type=str, default='us-west-2', help='region_name')
    parser.add_argument('--bucket', type=str, default='106839800180-23-07-17-15-02-02-bucket', help='output file')
    parser.add_argument('--aos_endpoint', type=str, default='vpc-domainc6g17c529-y8fodglwythm-5he24vsymfmfh5derkzrj4l4ry.us-west-2.es.amazonaws.com', help='Opensearch domain endpoint')
    parser.add_argument('--emb_model_endpoint', type=str, default='st-paraphrase-mpnet-node10-2023-07-30-16-39-58-975-endpoint', help='embedding model endpoint')
    parser.add_argument('--path_prefix', type=str, default='intention/2024-05-08/delete/', help='file path prefix')
    parser.add_argument('--index_name', type=str, default='chatbot-index-example-default-a', help='use this flag to specify the standby index')
    args = parser.parse_args()
    
    region = args.region
    s3 = boto3.client('s3', region)
    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, 'es', session_token=credentials.token)

    bucket = args.bucket
    aos_endpoint = args.aos_endpoint
    emb_model_endpoint = args.emb_model_endpoint
    path_prefix = args.path_prefix
    index_name = args.index_name

    file_generator = list_s3_objects(s3, bucket_name=bucket, prefix=path_prefix)
    for obj_key in file_generator:
        print(f"deleting intention example file - {obj_key}")
        success = delete_doc_index(aos_endpoint, obj_key, emb_model_endpoint, index_name)
        status = 'Successed' if success else 'Failed'
        print(f"deleting intention example file - {obj_key}, status : {status}")
        time.sleep(3)

    print(f"Finish {index_name}'s deletion")