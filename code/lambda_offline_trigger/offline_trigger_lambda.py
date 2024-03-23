import json
import boto3
import os
import datetime


JOBNAME = os.environ['glue_jobname']
embedding_endpoint = os.environ.get("embedding_endpoint", "")
def lambda_handler(event, context):

    glue = boto3.client('glue')

    bucket = event['Records'][0]['s3']['bucket']['name']
    object_key = event['Records'][0]['s3']['object']['key']
    company = object_key.split('/')[1] if len(object_key.split('/')) == 4 else 'default'
    publish_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
    
    print("**** in lambda bucket: " + bucket)
    print("**** in lambda object_key: " + object_key)
    print("**** in lambda publish_date: " + publish_date)
    print("**** in lambda company: " + company)

    glue.start_job_run(JobName=JOBNAME, Arguments={"--bucket": bucket, "--object_key": object_key, "--EMB_MODEL_ENDPOINT": embedding_endpoint, "--PUBLISH_DATE": publish_date, "--company" : company})

    return {
        'statusCode': 200,
        'body': json.dumps('Successful ')
    }
