import json
import boto3
import os

JOBNAME = os.environ['glue_jobname']
embedding_endpoint = os.environ.get("embedding_endpoint", "")
def lambda_handler(event, context):

    glue = boto3.client('glue')

    bucket = event['Records'][0]['s3']['bucket']['name']
    object_key = event['Records'][0]['s3']['object']['key']
    
    print("**** in lambda : " + bucket)
    print("**** in lambda : " + object_key)

    glue.start_job_run(JobName=JOBNAME, Arguments={"--bucket": bucket, "--object_key": object_key, "--EMB_MODEL_ENDPOINT": embedding_endpoint})

    return {
        'statusCode': 200,
        'body': json.dumps('Successful ')
    }
