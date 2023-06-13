#!/usr/bin/env python
# coding: utf-8

from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth, helpers
import boto3
import random
import json
from awsglue.utils import getResolvedOptions
import sys
import hashlib
import datetime
import re

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

def get_embedding(smr_client, text_arrs, endpoint_name=EMB_MODEL_ENDPOINT):
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
                    "inputs": text_arrs,
                    "parameters": parameters
                }
                ),
                ContentType="application/json",
            )
    
    json_str = response_model['Body'].read().decode('utf8')
    json_obj = json.loads(json_str)
    embeddings = json_obj["sentence_embeddings"]
    
    return embeddings

def iterate_paragraph(file_content, smr_client, index_name, endpoint_name):
    json_arr = json.loads(file_content)
    publish_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for idx, json_item in enumerate(json_arr):
        header = json_item['heading'][0]['heading']
        paragraph_content = json_item['content']

        if len(paragraph_content) > 1024:
            continue

        #whole paragraph embedding
        whole_paragraph_emb = get_embedding(smr_client, [paragraph_content,], endpoint_name)
        document = { "publish_date": publish_date, "idx":idx, "doc" : paragraph_content, "doc_type" : "Paragraph", "content" : paragraph_content, "doc_title": header, "doc_category": "", "embedding" : whole_paragraph_emb[0]}
        yield {"_index": index_name, "_source": document, "_id": hashlib.md5(str(document).encode('utf-8')).hexdigest()}
        
        # for every sentence
        sentences = re.split('[。？?.！!]', paragraph_content)
        sentences = [ sent for sent in sentences if len(sent) > 2 ]
        sentences_count = len(sentences)
        print("Process {} sentences in one batch".format(sentences_count))
        start = 0 
        while start < sentences_count:
            sentence_slices = sentences[start:start+20]
            print("Process {}-{} sentences in one micro-batch".format(start, start+20))
            start += 20
            embeddings = get_embedding(smr_client, sentence_slices, endpoint_name)
            for sent_id, sent in enumerate(sentence_slices):
                document = { "publish_date": publish_date, "idx":idx, "doc" : sent, "doc_type" : "Sentence", "content" : paragraph_content, "doc_title": header, "doc_category": "", "embedding" : embeddings[sent_id]}
                yield {"_index": index_name, "_source": document, "_id": hashlib.md5(str(document).encode('utf-8')).hexdigest()}

def iterate_QA(file_content, smr_client, index_name, endpoint_name):
    json_content = json.loads(file_content)
    json_arr = json_content["qa_list"]

    for json_item in json_arr:
        q = json_item['Question']
        a = json_item['Answer']

    doc_title = json_content["doc_title"]
    doc_category = json_content["doc_category"]
    questions = [ json_item['Question'] for json_item in json_arr ]
    answers = [ json_item['Answer'] for json_item in json_arr ]
    embeddings = get_embedding(smr_client, questions, endpoint_name)

    publish_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for i in range(len(embeddings)):
        document = { "publish_date": publish_date, "doc" : questions[i], "doc_type" : "Question", "content" : answers[i], "doc_title": doc_title, "doc_category": doc_category, "embedding" : embeddings[i]}
        yield {"_index": index_name, "_source": document, "_id": hashlib.md5(str(document).encode('utf-8')).hexdigest()}

def WriteVecIndexToAOS(file_content, content_type, smr_client, aos_endpoint=AOS_ENDPOINT, region=REGION, index_name=INDEX_NAME):
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

    gen_aos_record_func = None
    if content_type == "faq":
        gen_aos_record_func = iterate_QA(file_content, smr_client, index_name, EMB_MODEL_ENDPOINT)
    elif content_type == "paragraph":
        gen_aos_record_func = iterate_paragraph(file_content, smr_client, index_name, EMB_MODEL_ENDPOINT)
    else:
        raise RuntimeError('No Such Content type supported') 

    response = helpers.bulk(client, gen_aos_record_func)
    return response

def split_by(content, sep):
    return content.split(sep)

def process_s3_uploaded_file(bucket, object_key):
    print("********** object_key : " + object_key)
    obj = s3.Object(bucket,object_key)
    body = obj.get()['Body'].read().decode('utf-8').strip()

    if object_key.endswith(".faq"):
        print("********** pre-processing faq file")
        if(len(body) > 0):
            WriteVecIndexToAOS(body, "faq", smr_client)
    elif object_key.endswith(".txt"):
        print("********** pre-processing paragraph file")
        if(len(body) > 0):
            WriteVecIndexToAOS(body, "paragraph", smr_client)

    #print("********** body : " + body)
            

#if __name__ == '__main__':
#paragraph_array = ["Question: 在中国区是否可用？\nAnswer: 目前没有落地中国区的时间表，已经在以下区域推出：美国东部（弗吉尼亚州北部）、美国东部（俄亥俄州）、美国西部（俄勒冈州）、亚太地区（首尔）、亚太地区（新加坡）、亚太地区（悉尼）、亚太地区（东京）、欧洲地区（法兰克福）、欧洲地区（爱尔兰）、欧洲地区（伦敦）和欧洲地区（斯德哥尔摩）","Question: 目前可以支持什么数据源的接入？ \nAnswer: 目前只支持S3，其他数据源近期没有具体计划。"]
 
process_s3_uploaded_file(bucket, object_key)
