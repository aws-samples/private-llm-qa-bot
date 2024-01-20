import logging
from opensearchpy import OpenSearch, RequestsHttpConnection
import boto3
import os
import time
import pytz
import json
from datetime import datetime
from requests_aws4auth import AWS4Auth

logger = logging.getLogger()
logger.setLevel(logging.INFO)
lambda_client= boto3.client('lambda')
credentials = boto3.Session().get_credentials()
region = boto3.Session().region_name
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, 'es', session_token=credentials.token)
DOC_INDEX_TABLE= 'chatbot_doc_index'

def delete_doc_index(obj_key,embedding_model,index_name):
    def delete_aos_index(obj_key,index_name,size=50):
        aos_endpoint = os.environ.get("aos_endpoint", "")
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

    success = delete_aos_index(obj_key,index_name)

    if success:
        ##删除ddb里的索引
        dynamodb = boto3.client('dynamodb')
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

def list_doc_index ():
    dynamodb = boto3.client('dynamodb')
    scan_params = {
        'TableName': DOC_INDEX_TABLE,
        'Select': 'ALL_ATTRIBUTES',  # Return all attributes
    }
    try:
        response = dynamodb.scan(**scan_params)
        return response['Items']
    except Exception as e:
        logger.info(str(e))
        return []

def get_template(id):
    dynamodb = boto3.client('dynamodb')
    if id:
        params = {
            'TableName': os.environ.get('prompt_template_table'),
            'Key': {'id': {'S': id}},  # Return all attributes
        }
        try:
            response = dynamodb.get_item(**params)
            return response['Item']
        except Exception as e:
            logger.info(str(e))
            return None   
    else:
        params = {
            'TableName': os.environ.get('prompt_template_table'),
            'Select': 'ALL_ATTRIBUTES',  # Return all attributes
        }
        try:
            response = dynamodb.scan(**params)
            return response['Items']
        except Exception as e:
            logger.info(str(e))
            return []
        
def add_template(item):
    dynamodb = boto3.client('dynamodb')
    params = {
        'TableName': os.environ.get('prompt_template_table'),
        'Item': item,  
    }
    try:
        dynamodb.put_item(**params)
        return True
    except Exception as e:
        logger.info(str(e))
        return False

def delete_template(key):
    dynamodb = boto3.client('dynamodb')
    params = {
        'TableName': os.environ.get('prompt_template_table'),
        'Key': key,  
    }
    try:
        dynamodb.delete_item(**params)
        return True
    except Exception as e:
        logger.info(str(e))
        return False
    

## 1. write the feedback in logs and loaded to kinesis
## 2. if lambda_feedback is setup, then call lambda_feedback for other managment operations
def handle_feedback(event):
    method = event.get('method')
    
    ##invoke feedback lambda to store in ddb
    fn = os.environ.get('lambda_feedback')
    if method == 'post':
        results = True
        body = event.get('body')
        ## actions types: thumbs-up,thumbs-down,cancel-thumbs-up,cancel-thumbs-down
        timestamp = time.time()
        utc_datetime = datetime.utcfromtimestamp(timestamp)
        # Set the timezone to UTC+8
        utc8_timezone = pytz.timezone('Asia/Shanghai')
        datetime_utc8 = utc_datetime.replace(tzinfo=pytz.utc).astimezone(utc8_timezone)
        json_obj = {
                "opensearch_doc":  [], #for kiness firehose log subscription filter name
                "log_type":'feedback',
                "msgid":body.get('msgid'),
                "timestamp":str(datetime_utc8),
                "username":body.get('username'),
                "session_id":body.get('session_id'),
                "action":body.get('action'),
                "feedback":body.get('feedback')
            }
        json_obj_str = json.dumps(json_obj, ensure_ascii=False)
        logger.info(json_obj_str)

        json_obj = {**json_obj,**body,'method':method}
        if fn:
            response = lambda_client.invoke(
                    FunctionName = fn,
                    InvocationType='RequestResponse',
                    Payload=json.dumps(json_obj)
                )
            payload_json = json.loads(response.get('Payload').read())
            logger.info(payload_json)
            results = payload_json['body']
            if response['StatusCode'] != 200 or not results:
                logger.info(f"invoke lambda feedback StatusCode:{response['StatusCode']} and result {results}")
                results = False
        return results
    elif method == 'get':
        results = []
        body = event.get('body')
        json_obj = {**body,'method':method}
        if fn:
            response = lambda_client.invoke(
                    FunctionName = fn,
                    InvocationType='RequestResponse',
                    Payload=json.dumps(json_obj)
                )
            if response['StatusCode'] == 200:
                payload_json = json.loads(response.get('Payload').read())
                results = payload_json['body']
        return results   
    elif method == 'delete':  
        results = True
        body = event.get('body')
        json_obj = {**body,'method':method}
        if fn:
            response = lambda_client.invoke(
                    FunctionName = fn,
                    InvocationType='RequestResponse',
                    Payload=json.dumps(json_obj)
                )
            if response['StatusCode'] == 200:
                payload_json = json.loads(response.get('Payload').read())
                results = payload_json['body']
        return results  
    
    
def management_api(method,resource,event):
    if method == 'delete' and resource == 'docs':
        logger.info(f"delete doc index of:{event.get('filename')}/{event.get('embedding_model')}/{event.get('index_name')}")
        delete_doc_index(event.get('filename'),event.get('embedding_model'),event.get('index_name'))
        return {'statusCode': 200}
    ## 如果是get doc index操作
    elif method == 'get' and resource == 'docs':
        results = list_doc_index()
        return {'statusCode': 200,'body':results }
    ## 如果是get template 操作
    elif method == 'get' and resource == 'template':
        id = event.get('id')
        results = get_template(id)
        return {'statusCode': 200,'body': {} if results is None else results }
    ## 如果是add a template 操作
    elif method == 'post' and resource == 'template':
        body = event.get('body')
        item = {
            'id': {'S': body.get('id')},
            'template_name':{'S':body.get('template_name','')},
            'template':{'S':body.get('template','')},
            'comment':{'S':body.get('comment','')},
            'username':{'S':body.get('username','')}
        }
        result = add_template(item)
        return {'statusCode': 200 if result else 500,'body':result }
     ## 如果是delete a template 操作
    elif method == 'delete' and resource == 'template':
        body = event.get('body')
        key = {
            'id': {'S': body.get('id')}
        }
        result = delete_template(key)
        return {'statusCode': 200 if result else 500,'body':result }

    ## 处理feedback action
    elif method in ['post','get','delete'] and resource == 'feedback':
        results = handle_feedback(event)
        return {'statusCode': 200 if results else 500,'body':results}
    
    else:
        logger.info(f'not supported api {resource}/{method}')
        return {'statusCode': 400 ,'body':f'not supported api {resource}/{method}'}