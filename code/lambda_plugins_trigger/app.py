import json
import boto3

import requests
from requests_aws4auth import AWS4Auth

credentials = boto3.Session().get_credentials()
region = boto3.Session().region_name
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, 'lambda', session_token=credentials.token)


##for test
def lambda_handler(event, context):
    url = 'https://gbs47wyp3itmmfcta2w5bc674y0hnrae.lambda-url.us-east-2.on.aws/todos/river'
    headers={"Content-Type": "application/json"}
    data = {'todo': event.get('todo')}
    resp = requests.post(url, json = data, headers=headers,auth=awsauth)
    # resp = requests.get(url,auth=awsauth)
    print(resp.text)
    return {
        'statusCode': 200,
        'body': json.dumps('success')
    }
