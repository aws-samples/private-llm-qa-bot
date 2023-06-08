#!/usr/bin/env python
# coding: utf-8

from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth, helpers
import boto3
import random
import json
from awsglue.utils import getResolvedOptions
import sys
import hashlib

args = getResolvedOptions(sys.argv, ['bucket', 'object_key','AOS_ENDPOINT','REGION','EMB_MODEL_ENDPOINT'])
s3 = boto3.resource('s3')
bucket = args['bucket']
object_key = args['object_key']

# EMB_MODEL_ENDPOINT = "st-paraphrase-mpnet-base-v2-2023-04-19-04-14-31-658-endpoint"
EMB_MODEL_ENDPOINT=args['EMB_MODEL_ENDPOINT']
smr_client = boto3.client("sagemaker-runtime")

# AOS_ENDPOINT = 'vpc-chatbot-knn-3qe6mdpowjf3cklpj5c4q2blou.us-east-1.es.amazonaws.com'
AOS_ENDPOINT = args['AOS_ENDPOINT']
REGION = args['REGION']
INDEX_NAME = 'chatbot-index'
# REGION='us-east-1'

def get_st_embedding(smr_client, text_input, endpoint_name=EMB_MODEL_ENDPOINT):
    parameters = {
      #"early_stopping": True,
      #"length_penalty": 2.0,
      "max_new_tokens": 50,
      "temperature": 0,
      "min_length": 10,
      "no_repeat_ngram_size": 2,
    }

    response_model = smr_client.invoke_endpoint(
                EndpointName=endpoint_name,
                Body=json.dumps(
                {
                    "inputs": [text_input],
                    "parameters": parameters
                }
                ),
                ContentType="application/json",
            )
    
    json_str = response_model['Body'].read().decode('utf8')
    json_obj = json.loads(json_str)
    embeddings = json_obj["sentence_embeddings"]
    
    return embeddings[0]


def WriteVecIndexToAOS(paragraph_array, smr_client, aos_endpoint=AOS_ENDPOINT, region=REGION, index_name=INDEX_NAME):
    """
    write paragraph to AOS for Knn indexing.
    :param paragraph_input : document content 
    :param aos_endpoint : AOS endpoint
    :param index_name : AOS index name
    :return None
    """
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, region)
    # auth = ('xxxx', 'yyyy') master user/pwd
    # auth = (aos_master, aos_pwd)

    client = OpenSearch(
        hosts = [{'host': aos_endpoint, 'port': 443}],
        http_auth = auth,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection
    )
    
    def get_embs():
        for paragraph in paragraph_array:
            print("********** paragraph : " + paragraph)

            if len(paragraph) < 10:
                continue

            documents = []
            if paragraph.lower().find("question:") > -1:
                question, answer = paragraph.strip().split("\n", 1)
                question = question.replace("Question: ", "")
                answer = answer.replace("Answer: ", "")
                documents.append({ "doc" : question, "doc_type" : "Q", "answer" : answer, "embedding" : get_st_embedding(smr_client, question)})
            else:
                documents.append({ "doc" : paragraph, "doc_type" : "P", "answer" : "", "embedding" : get_st_embedding(smr_client, paragraph)})

            for document in documents:
                yield {"_index": index_name, "_source": document, "_id": hashlib.md5(str(document).encode('utf-8')).hexdigest()}

    get_embs_func = get_embs()

    response = helpers.bulk(client, get_embs_func)
    return response


def split_by(content, sep):
    return content.split(sep)

def process_s3_uploaded_file(bucket, object_key):
    
    print("********** object_key : " + object_key)
 
    sep = '====='
    if object_key.endswith(".faq"):
        print("********** pre-processing faq file")
    elif object_key.endswith(".txt"):
        print("********** pre-processing paragraph file")

    obj = s3.Object(bucket,object_key)
    body = obj.get()['Body'].read().decode('utf-8').strip()
    #print("********** body : " + body)
            
    if(len(body) > 0):
        WriteVecIndexToAOS(split_by(body, sep), smr_client)

#if __name__ == '__main__':
#paragraph_array = ["Question: 在中国区是否可用？\nAnswer: 目前没有落地中国区的时间表，已经在以下区域推出：美国东部（弗吉尼亚州北部）、美国东部（俄亥俄州）、美国西部（俄勒冈州）、亚太地区（首尔）、亚太地区（新加坡）、亚太地区（悉尼）、亚太地区（东京）、欧洲地区（法兰克福）、欧洲地区（爱尔兰）、欧洲地区（伦敦）和欧洲地区（斯德哥尔摩）","Question: 目前可以支持什么数据源的接入？ \nAnswer: 目前只支持S3，其他数据源近期没有具体计划。"]
 
process_s3_uploaded_file(bucket, object_key)
