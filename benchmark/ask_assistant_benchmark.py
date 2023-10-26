#!/usr/bin/env python
# coding: utf-8
import argparse
import time
import json
import numpy as np
import random
from boto3 import client as boto3_client

lambda_client = boto3_client('lambda')

def call_bedrock(llm_model_name, embedding_model_endpoint):
    """
    llm_model_name could be 'claude' or 'claude-instant'
    
    """
    queries = ["AWS Clean Rooms的数据源可以支持哪些？", "Clean Rooms能查多大规模数据", "Clean Rooms 如何计费", "clean Rooms在中国区可用吗", "Clean Rooms支持哪些数据源"]
    prompt = random.choice(queries)

    msg = {
        "OPENAI_API_KEY":"",
        "ws_endpoint":"",
        "msgid":"id-1697113325054-a9a2c2c83f75_res",
        "chat_name":"MsChGelFvHcCF9g=",
        "prompt":prompt,
        "model":llm_model_name,
        "use_qa":True,
        "imgurl":"",
        "template_id":"default",
        "max_tokens":2000,
        "temperature":0.01,
        "system_role":"AWSBot",
        "system_role_prompt":"你是云服务AWS的智能客服机器人",
        "embedding_model": embedding_model_endpoint
    }

    start = time.time()
    invoke_response = lambda_client.invoke(FunctionName="Ask_Assistant",
                                               InvocationType='RequestResponse',
                                               Payload=json.dumps(msg))
    end = time.time()
    payload_json = json.loads(invoke_response.get('Payload').read())  
    print(payload_json['body'][0]['choices'][0]['text'])
    return end - start

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat_num', type=int, default=100, help='run times')
    parser.add_argument('--llm_model_name', type=str, default='claude', help='input model name')
    parser.add_argument('--embedding_model_endpoint', type=str, default='bge-zh-15-2023-09-17-01-00-27-086-endpoint', help='embedding endpoint')

    args = parser.parse_args()
    repeat_num = args.repeat_num
    llm_model_name = args.llm_model_name
    embedding_model_endpoint = args.embedding_model_endpoint
    
    elapse_times = []
    for i in range(repeat_num):
        duration = call_bedrock(llm_model_name, embedding_model_endpoint)
        print("{}-th : {}".format(i, duration))
        elapse_times.append(duration)
    
    mean = np.mean(elapse_times)
    variance = np.var(elapse_times)
    median = np.median(elapse_times)
    p75 = np.percentile(elapse_times, 75)
    p90 = np.percentile(elapse_times, 90)
    p99 = np.percentile(elapse_times, 99)
    print("\n\n---------[Stat]----------\n\n Mean: {}, Variance: {}, p50,p75,p90,p99: [{},{},{},{}]".format(mean, variance, median, p75, p90, p99))