import json
import logging
import os
import boto3
from enum import Enum
import tempfile
logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb_client = boto3.resource('dynamodb')
user_feedback_table = os.environ.get('user_feedback_table')
chat_session_table = os.environ.get('chat_session_table')

## content  = [question, answer, intention,msgid]
def get_session_by_msgid(session_id,msg_id,user_id):

    # table name
    table = dynamodb_client.Table(chat_session_table)
    operation_result = []
    try:
        response = table.get_item(Key={'session-id': session_id,'user_id':user_id})
        if "Item" in response.keys():
            operation_result = json.loads(response["Item"]["content"])
            operation_result = [(item[0],item[1],item[3]) for item in operation_result if item[3] == msg_id]
    except Exception as e:
        logger.info(f"get session failed {str(e)}")
    return operation_result

## content  = [question, answer, intention,msgid]
def get_qa_by_msgid(session_id,msg_id):

    # table name
    table = dynamodb_client.Table(user_feedback_table)
    operation_result = []
    try:
        response = table.get_item(Key={'session-id': session_id,'msgid':msg_id})
        if "Item" in response.keys():
            operation_result = json.loads(response["Item"]["content"])
    except Exception as e:
        logger.info(f"get session failed {str(e)}")
    return operation_result


## update feedback table 
def update_qa_status(session_id,msg_id,status):
    table = dynamodb_client.Table(user_feedback_table)
    operation_result = False
    try:
        response = table.get_item(Key={'session-id': session_id,'msgid':msg_id})
        if "Item" in response.keys():
            result = json.loads(response["Item"]["content"])
            content = json.dumps([{**result[0],"action":status}],ensure_ascii=False)

            response2 = table.put_item(
                Item={
                    'session-id': session_id,
                    'msgid':msg_id,
                    'content': content
                }
            )
            if "ResponseMetadata" in response2.keys():
                if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
                    operation_result = True
                else:
                    operation_result = False
            else:
                operation_result = False

    except Exception as e:
        logger.info(f"get session failed {str(e)}")
    return operation_result




def update_feedback(session_id,msgid,action,username,timestamp,feedback=''):

    chat_data = get_session_by_msgid(session_id,msgid,username)

    if not chat_data:
        print('No chat data found')
        return False 
    
    question, answer = chat_data[0][0],chat_data[0][1]

    table = dynamodb_client.Table(user_feedback_table)
    operation_result = True

    response = table.get_item(Key={'session-id': session_id,"msgid":msgid})

    # if "Item" in response.keys():
    #     chat_history = json.loads(response["Item"]["content"])
    # else:
    #     # print("****** No result")
        # chat_history = []
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
    operation_result = True
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
    
    
##add new qa pair     
def add_new_qa(session_id,msgid,action,username,timestamp,question,answer):
    table = dynamodb_client.Table(user_feedback_table)
    chat_history = []
    feedback = {
        "question":question,
        "answer":answer,
        "username":username,
        "action":action,
        "timestamp":timestamp,
        "feedback":answer
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
    operation_result = True
    if "ResponseMetadata" in response.keys():
        if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            operation_result = True
        else:
            operation_result = False
    else:
        operation_result = False
    return operation_result


## save to s3 bucket to trigger glue job
def save_string_to_s3_bucket(text_string, bucket_name, file_name,s3_prefix=""):
    s3 = boto3.client('s3')
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file_name = temp_file.name

    s3_key = os.path.join(s3_prefix, file_name)
    try:
        temp_file.write(text_string.encode('utf-8'))
        temp_file.close()
        s3.upload_file(temp_file_name,bucket_name,s3_key,ExtraArgs={'Metadata': {"category":"UGC"}})
        logger.info(f"uploaded file to:{bucket_name}/{s3_key}")
        if os.path.exists(temp_file_name):
            os.remove(temp_file_name)
        return True
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        logger.error(f"faild to upload file to:{bucket_name}/{s3_key}")
        if os.path.exists(file_name):
            os.remove(file_name)
        return False

def delete_qa(session_id,msgid):
    table = dynamodb_client.Table(user_feedback_table)
    try:
        table.delete_item(
        Key={
            'session-id': session_id,"msgid":msgid
        })
        return True
    except Exception as e:
        logger.error(f"delete session failed {str(e)}")
        return False




 ##把faq知识入库，从ddb中取出faq，写入到知识库中
def inject_new_qa(session_id,msgid,bucket_name,s3_prefix,action):
    chat_data = get_qa_by_msgid(session_id,msgid)
    if not chat_data:
        logger.error('No chat data found')
        return False

    question, answer,username = chat_data[0]['question'],chat_data[0]['feedback'],chat_data[0]['username']
    formatted_qa = 'Question: {}\nAnswer: {}\n'.format(question,answer)
    logger.info(f"formatted_qa:{formatted_qa}")
    bucket_name = os.environ.get('UPLOAD_BUCKET') if bucket_name == '' or bucket_name == None else bucket_name
    s3_prefix = os.environ.get('UPLOAD_OBJ_PREFIX') if s3_prefix == '' or s3_prefix == None else s3_prefix

    temp_filename = f'{msgid}_{username}.faq'
    operation_result = save_string_to_s3_bucket(
        text_string=formatted_qa,
        bucket_name = bucket_name,
        file_name = temp_filename,
        s3_prefix = f'{s3_prefix}{username}/'
    )
    if operation_result:
        operation_result = update_qa_status(session_id,msgid,status=action)

    return operation_result

def lambda_handler(event, context):
    logger.info(f"event:{event}")
    method = event.get('method')
    
    ##增加或者修改反馈
    if method == 'post':
        session_id = event.get('session_id')
        msgid = event.get('msgid')
        action = event.get('action')
    
        username = event.get('username')
        timestamp = event.get('timestamp')
        feedback =  event.get('feedback')
        question = event.get('question')
        answer = event.get('answer')
        
        ret = True
        ##增加一个FAQ，保存到ddb
        if action == 'new-added':
            ret = add_new_qa(session_id,msgid,action,username,timestamp,question,answer)

        ##把faq知识入库，从ddb中取出faq，写入到知识库中
        elif action == 'injected':
            bucket_name = event.get('s3_bucket')
            s3_prefix = event.get('obj_prefix')
            ret = inject_new_qa(session_id,msgid,bucket_name,s3_prefix,action)

        else:
            ret = update_feedback(session_id,msgid,action,username,timestamp,feedback)
        return {
            'statusCode': 200,
            'body':ret
        }
        
    ## 获取反馈
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
    ## 删除反馈
    elif method == 'delete':
        session_id = event.get('session_id')
        msgid = event.get('msgid')
        ret = delete_qa(session_id,msgid)
        return {
            'statusCode': 200,
            'body':ret
        }

