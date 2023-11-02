#!/usr/bin/env python
# coding: utf-8
import argparse
import time
import json
import numpy as np
import random
from boto3 import client as boto3_client

lambda_client = boto3_client('lambda')

def call_bedrock(query, model_name):
    msg = {
      "fewshot_cnt": 5,
      "query": query,
      "use_bedrock" : "True",
      "llm_model_name" : model_name
    }

    start = time.time()
    invoke_response = lambda_client.invoke(FunctionName="Detect_Intention",
                                               InvocationType='RequestResponse',
                                               Payload=json.dumps(msg))
    end = time.time()
    payload_json = json.loads(invoke_response.get('Payload').read())  
    print("{}=>{}".format(query, payload_json))
    return end - start

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat_num', type=int, default=100, help='run times')
    parser.add_argument('--llm_model_name', type=str, default='claude', help='it could be claude or claude-instant')

    args = parser.parse_args()
    repeat_num = args.repeat_num
    llm_model_name = args.llm_model_name
    queries = [
        "Baywatch里Unoccupied hosts是什么意思？",
        "Baywatch里Free slots是什么意思？",
        "baywatch是什么?",
        "如何知道一个实例类型是否可用了？",
        "什么是foob？",
        "有哪些foob的例子？",
        "在哪个场景下，我需要foob？",
        "如果客户有对EBS的需求该怎样？",
        "Incremental Capacity Request ICR ticket和Limit increase的区别是啥？",
        "如何申请p4d, p5？",
        "申请p4d的模版是什么样的？",
        "ODCR预留怎么计费？",
        "ODCR怎么享受RI 或者Savings Plan计费？",
        "what is baywatch?",
        "怎么看现有的Capacity？",
        "申请gpu foob的流程是什么？",
        "baywatch中的Unoccupied hosts是什么？",
        "baywatch中的Free slots 是啥？",
        "Amazon EMR有哪些优势？",
        "who is the GTMS of quicksight?",
        "who is the product manager of quicksight?",
        "who is the sales of AIML in north?",
        "who is the BD of AIML in north?",
        "Who should I contact for questions related to Sagemaker?",
        "Who should I contact for questions related to EMR?"
    ]
    
    elapse_times = []
    for i in range(repeat_num):
        for j, query in enumerate(queries):
            duration = call_bedrock(query, llm_model_name)
            print("{}-iteration {}-th : {}".format(i, j, duration))
            elapse_times.append(duration)
    
    mean = np.mean(elapse_times)
    variance = np.var(elapse_times)
    median = np.median(elapse_times)
    p75 = np.percentile(elapse_times, 75)
    p90 = np.percentile(elapse_times, 90)
    p99 = np.percentile(elapse_times, 99)
    print("\n\n---------[Stat]----------\n\n Mean: {}, Variance: {}, p50,p75,p90,p99: [{},{},{},{}]".format(mean, variance, median, p75, p90, p99))