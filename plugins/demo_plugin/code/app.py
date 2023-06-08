import json,re
import boto3
import os

dynamodb = boto3.resource('dynamodb')
table_name = os.environ.get('TABLE_NAME')
ddb_table = dynamodb.Table(table_name)


#增加一个记录
def addItem(key_name,item):
    origin = ddb_table.get_item(Key={'username':key_name})
    print('get origin:',origin)
    
    if not origin.get('Item'):
        todos = [item]
    else:
        origin_todos = origin['Item']['todos']
        todos = json.loads(origin_todos)+[item]        
    ddb_table.put_item(Item={
        'username':key_name,
        'todos':json.dumps(todos)
    })    

#遍历出所有的记录
def listAllItems(key_name = None):
    results = []
    key_name = key_name if key_name else 'global'
    response = ddb_table.scan(
            FilterExpression=f"#k = :val",
            ExpressionAttributeNames={"#k": 'username'},
            ExpressionAttributeValues={":val": key_name}
    )
    for item in response['Items']:
        results.append(item)
    while 'LastEvaluatedKey' in response:
        response = ddb_table.scan(
            FilterExpression=f"#k = :val",
            ExpressionAttributeNames={"#k": 'username'},
            ExpressionAttributeValues={":val": key_name},
            ExclusiveStartKey=response['LastEvaluatedKey']
            )
        for item in response['Items']:
            results.append(item)
   
    return results

def lambda_handler(event, context):
    method = event['requestContext']['http']['method']
    path = event['requestContext']['http']['path']
    print(f'method:{method},path:{path}')
    path_matched_group = re.match(r'(/\w+)/?(\w+)?',path) # /path/{username}
    
    #GET 方法
    if method == 'GET' and path_matched_group.group(1) == '/todos':
        username = path_matched_group.group(2)
        resp = listAllItems(username)
        return {
        'statusCode': 200,
        'body': json.dumps(resp)
        }
        
    #POST 方法
    elif method == 'POST' and path_matched_group.group(1) == '/todos':
        username = path_matched_group.group(2)
        
        #如果没有用username，则用global name
        if not username:
            username = 'global'
        body = json.loads(event['body'])
        print(body)
        addItem(username,body.get('todo'))
        return {
        'statusCode': 200,
        'body': json.dumps('add todo list success')
        }




    
   

