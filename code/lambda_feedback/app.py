import json
import logging
import time
import os
import re
from datetime import datetime
import boto3
import time
import hashlib
import uuid
from enum import Enum

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb_client = boto3.resource('dynamodb')
user_feedback_table = os.environ.get('user_feedback_table')
chat_session_table = os.environ.get('chat_session_table')


## content  = [question, answer, intention,msgid]
def get_session_by_msgid(session_id,msg_id):

    table_name = chat_session_table

    # table name
    table = dynamodb_client.Table(table_name)
    operation_result = []
    try:
        response = table.get_item(Key={'session-id': session_id})
        if "Item" in response.keys():
            operation_result = json.loads(response["Item"]["content"])
            operation_result = [(item[0],item[1],item[3]) for item in operation_result if item[3] == msg_id]
    except Exception as e:
        logger.info(f"get session failed {str(e)}")
    return operation_result



def update_feedback(session_id,msgid,action,username,timestamp,feedback=''):

    chat_data = get_session_by_msgid(session_id,msgid)

    if not chat_data:
        print('No chat data found')
        return True 
    
    question, answer = chat_data[0][0],chat_data[0][1]

    table = dynamodb_client.Table(user_feedback_table)
    operation_result = True

    response = table.get_item(Key={'session-id': session_id,"msgid":msgid})

    if "Item" in response.keys():
        chat_history = json.loads(response["Item"]["content"])
    else:
        # print("****** No result")
        chat_history = []

    feedback = {
        "question":question,
        "answer":answer,
        "username":username,
        "action":action,
        "timestamp":timestamp,
        "feedback":feedback
    }
    chat_history.append(feedback)
    content = json.dumps(chat_history,ensure_ascii=False)

    # inserting values into table
    response = table.put_item(
        Item={
            'session-id': session_id,
            'msgid':msgid,
            'content': content
        }
    )

    if "ResponseMetadata" in response.keys():
        if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            operation_result = True
        else:
            operation_result = False
    else:
        operation_result = False

    return operation_result

def get_feedback(pageindex,textfilter,pagesize,filteringTokens,filteringOperation):
    table = dynamodb_client.Table(user_feedback_table)
    items = []
    cnt_resp = table.scan(Select='COUNT')
    total_cnt = cnt_resp['Count']
    
    exclusive_start_key = pageindex
    if exclusive_start_key:
        response = table.scan(Limit=pagesize,ExclusiveStartKey=exclusive_start_key)
    else:
        response = table.scan(Limit=pagesize)
    if 'LastEvaluatedKey' in response:
        exclusive_start_key = response['LastEvaluatedKey']
    else:
        exclusive_start_key = None

    items = response['Items']
    items = filteringActionResults(items,filteringTokens,filteringOperation)
    
    return items,exclusive_start_key,total_cnt

def filteringActionResults(items,filteringTokens,filteringOperation):
    print('filteringTokens:',filteringTokens)

    tokens  = json.loads(filteringTokens)
    if not tokens:
        return items
    propertyKey = tokens[0].get('propertyKey')
    if propertyKey != 'action':
        return items
    value =  tokens[0].get('value')
    new_items = [it for it in items if json.loads(it['content'])[-1]['action'] == value]
    print(new_items)
    return new_items
    
    

def lambda_handler(event, context):
    logger.info(f"event:{event}")
    method = event.get('method')
    if method == 'post':
        session_id = event.get('session_id')
        msgid = event.get('msgid')
        action = event.get('action')
        username = event.get('username')
        timestamp = event.get('timestamp')
        feedback =  event.get('feedback')
        ret = update_feedback(session_id,msgid,action,username,timestamp,feedback)
        return {
            'statusCode': 200 if ret else 500
        }
    elif method == 'get':
        
        pageindex_key = None if event.get('pageindex_key') == 'undefined' else json.loads(event.get('pageindex_key'))
        textfilter = event.get('textfilter')
        pagesize = int(event.get('pagesize'))
        filteringTokens = event.get('filteringTokens')
        filteringOperation = event.get('filteringOperation')
        items,exclusive_start_key,total_cnt = get_feedback(pageindex_key,textfilter,pagesize,filteringTokens,filteringOperation)
        return {
            'statusCode': 200,
            'body':{
                "items":items,
                "total_cnt":total_cnt,
                "pageindex_key":exclusive_start_key
            }
        }
