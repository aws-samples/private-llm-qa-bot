#!/usr/bin/env python
# coding: utf-8
'''
Initialize AOS Index, 在cloud9 上运行成功，但仅仅当cloud9外网IP开放时 才能执行。否则报错
'{"Message":"User: anonymous is not authorized to perform: es:ESHttpPut because no resource-based policy allows the es:ESHttpPut action"}'
'''
import sys
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from awsglue.utils import getResolvedOptions

AOS_ENDPOINT = 'search-chatbot-workshop-r5rtwkjmtfuc5xz2vv5ychwfva.us-east-1.es.amazonaws.com'
INDEX_NAME = 'chatbot-index'
REGION='us-east-1'

args = getResolvedOptions(sys.argv, ['aos_master', 'aos_password'])
aos_master = args['aos_master']
aos_pwd = args['aos_password']
auth = (aos_master, aos_pwd)

client = OpenSearch(
    hosts = [{'host': AOS_ENDPOINT, 'port': 443}],
    http_auth = auth,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection
)

index_body = {
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
            "doc": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart"
            },
            "embedding": {
                "type": "knn_vector",
                "dimension": 768,
                "method": {
                    "name": "hnsw",
                    "space_type": "l2",
                    "engine": "nmslib",
                    "parameters": {
                        "ef_construction": 64,
                        "m": 8
                    }
                }            
            }
        }
    }
}

response = client.indices.create(INDEX_NAME, body=index_body)
print(response)

# Dashboard command
# 1. Create AOS index
'''
PUT chatbot-index
{
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
            "doc_type" : {
                "type" : "keyword"
            },
            "doc": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart"
            },
            "embedding": {
                "type": "knn_vector",
                "dimension": 768,
                "method": {
                    "name": "hnsw",
                    "space_type": "l2",
                    "engine": "nmslib",
                    "parameters": {
                        "ef_construction": 512,
                        "m": 32
                    }
                }            
            }
        }
    }
}
'''
# 2. Search AOS with term condition (term field should be keyword, not text)
'''
GET chatbot-index/_search
{
    "size": 1,
    "query": {
      "bool":{
        "must":[ {"term": { "doc_type":"P" }} ],
        "should": [ {"match": { "doc": "强化部件" }} ]
      }
    },
    "sort": [
      {
        "_score": {
          "order": "desc"
        }
      }
    ]
}
'''
