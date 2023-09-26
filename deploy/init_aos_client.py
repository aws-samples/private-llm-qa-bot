#!/usr/bin/env python
# coding: utf-8

import os
import sys
import boto3
import json
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

def create_aos_index(aos_endpoint, index_name, index_definition):
    sess = boto3.Session()
    credentials = sess.get_credentials()
    current_region = sess.region_name

    auth = AWSV4SignerAuth(credentials, current_region)

    client = OpenSearch(
        hosts = [{'host': aos_endpoint, 'port': 443}],
        http_auth = auth,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection
    )
    
    response = client.indices.create(index=index_name, body=index_definition)
    if response.get('acknowledged'):
        print(f"Index {index_name} created successfully")
    else:
        print(f"Failed to create index {index_name}")

aos_endpoint = sys.argv[1] # "vpc-domain66ac69e0-7nk4nwvargjd-34nala5yssppv2ssbyl5ifco2y.us-east-1.es.amazonaws.com"

index_name = 'chatbot-index'
index_definition ={
    "settings" : {
        "index":{
            "number_of_shards" : 1,
            "number_of_replicas" : 0,
            "knn": "true",
            "knn.algo_param.ef_search": 32
        }
    },
    "mappings": {
        "properties": {
            "publish_date" : {
                "type": "date",
                "format": "yyyy-MM-dd HH:mm:ss"
            },
            "idx" : {
                "type": "integer"
            },
            "doc_type" : {
                "type" : "keyword"
            },
            "doc": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart"
            },
            "content": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart"
            },
            "doc_title": {
                "type": "keyword"
            },
            "doc_category": {
                "type": "keyword"
            },
            "embedding": {
                "type": "knn_vector",
                "dimension": 1024,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",
                    "parameters": {
                        "ef_construction": 128,
                        "m": 16
                    }
                }            
            }
        }
    }
}

create_aos_index(aos_endpoint, index_name, index_definition)
