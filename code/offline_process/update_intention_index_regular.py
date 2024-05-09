import logging
import boto3
import os
import time
import pytz
import sys
import json
import math
import argparse
import itertools
from datetime import datetime
from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection
from awsglue.utils import getResolvedOptions

logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
log_format = '%(asctime)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

DOC_INDEX_TABLE= 'chatbot_doc_index'

args = getResolvedOptions(sys.argv, ['bucket', 'region','aos_endpoint','emb_model_endpoint','path_prefix', 'ssm_key_for_index_status', 'concurrent_runs_quota', 'job_name', 'company', 'emb_batch_size'])
region = args['region']

credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, 'es', session_token=credentials.token)

bucket = args['bucket']
aos_endpoint = args['aos_endpoint']
emb_model_endpoint = args['emb_model_endpoint']
path_prefix = args['path_prefix']
ssm_key_for_index_status = args['ssm_key_for_index_status']
# index_name = args['index_name'] # index_name should be chatbot-example-index-default-a or chatbot-example-index-default-b

concurrent_runs_quota = int(args['concurrent_runs_quota'])
job_name = args['job_name']
company = args['company']
emb_batch_size = args['emb_batch_size']

glue = boto3.client('glue', region)

def list_s3_objects(s3_client,bucket_name, prefix=''):
    objects = []
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    # iterate over pages
    for page in page_iterator:
        # loop through objects in page
        if 'Contents' in page:
            for obj in page['Contents']:
                yield obj['Key']
        # if there are more pages to fetch, continue
        if 'NextContinuationToken' in page:
            page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix,
                                                ContinuationToken=page['NextContinuationToken'])

def delete_doc_index(aos_endpoint, obj_key, embedding_model, index_name):
    global region
    def delete_aos_index(aos_endpoint, obj_key, index_name, size=50):
        global awsauth
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

    success = delete_aos_index(aos_endpoint, obj_key, index_name)

    if success:
        ##删除ddb里的索引
        dynamodb = boto3.client('dynamodb', region)
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
            return False

    return success

def update_running_job_set(job_name, running_job_id_set):
    global glue
    response = glue.get_job_runs(JobName=job_name)

    # Filter the finished job runs
    finish_runs_id = [run['Id'] for run in response['JobRuns'] if run['JobRunState'] in ['STOPPED', 'SUCCEEDED', 'FAILED', 'ERROR', 'TIMEOUT'] ]
    
    return running_job_id_set - set(finish_runs_id)

def count_s3_files(s3_client, bucket_name, prefix):
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    file_count = 0
    for page in page_iterator:
        for obj in page.get('Contents', []):
            if not obj['Key'].endswith('/'):
                file_count += 1
    return file_count


def start_job(glue_client, job_name, key_path, aos_endpoint, emb_model_endpoint, bucket, region_name, publish_date, company, emb_batch_size, index_name):
    global glue
    logger.info('start job for {}'.format(key_path))   
    response = glue.start_job_run(
        JobName=job_name,
        Arguments={
            '--additional-python-modules': 'pdfminer.six==20221105,gremlinpython==3.6.3,langchain==0.0.162,beautifulsoup4==4.12.2,boto3>=1.28.52,botocore>=1.31.52,,anthropic_bedrock,python-docx',
            '--object_key': key_path,
            '--REGION': region_name,
            '--AOS_ENDPOINT': aos_endpoint,
            '--bucket' : bucket,
            '--EMB_MODEL_ENDPOINT': emb_model_endpoint,
            '--PUBLISH_DATE': publish_date,
            '--company' : company,
            '--emb_batch_size' : str(emb_batch_size),
            '--index_name':index_name
            })  
    return response['JobRunId']

def batch_generator(generator, batch_size):
    while True:
        batch = list(itertools.islice(generator, batch_size))
        if not batch:
            break
        yield batch

def delete_intentions(region, bucket, path_prefix, aos_endpoint, emb_model_endpoint, index_name):
    s3 = boto3.client('s3', region)
    file_generator = list_s3_objects(s3, bucket_name=bucket, prefix=path_prefix)
    for obj_key in file_generator:
        logger.info(f"deleting intention example file - {obj_key}")
        success = delete_doc_index(aos_endpoint, obj_key, emb_model_endpoint, index_name)
        status = 'Successed' if success else 'Failed'
        logger.info(f"deleting intention example file - {obj_key}, status : {status}")
        time.sleep(3)

def add_intentions(region, bucket, path_prefix, aos_endpoint, emb_model_endpoint, index_name, concurrent_runs_quota, job_name, company, emb_batch_size):
    running_job_id_set = set()
    s3 = boto3.client('s3', region)
    file_cnt = count_s3_files(s3, bucket_name=bucket, prefix=path_prefix)
    batch_size = math.ceil(file_cnt / concurrent_runs_quota)
    logger.info(f"file_cnt: {file_cnt}, batch_size:{batch_size}")
    
    file_generator = list_s3_objects(s3, bucket_name=bucket, prefix=path_prefix)
    
    batches = batch_generator(file_generator, batch_size)
    publish_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 

    for idx, batch in enumerate(batches):
        key_list_str = ','.join(batch)
        if not key_list_str.endswith('.example'):
            logger.info("[{}] skip for {}, because it's not example file".format(idx, key_list_str))
            continue

        while len(running_job_id_set) >= concurrent_runs_quota:
            logger.info('wait for previous running job done')
            time.sleep(100)
            running_job_id_set = update_running_job_set(job_name, running_job_id_set)
            logger.info('concurrent_running: {}'.format(running_job_id_set))
            
        running_job_id=start_job(glue, job_name, key_list_str, aos_endpoint, emb_model_endpoint, bucket, region, publish_date, company, emb_batch_size, index_name)
        running_job_id_set.add(running_job_id)
        # sleep_seconds = len(running_job_id_set) * 2
        logger.info("[{}] running job count: {}".format(idx, len(running_job_id_set)))
        time.sleep(15)

    while len(running_job_id_set) > 0:
        logger.info("waiting all job finished, current running job: {}".format(str(running_job_id_set)))
        running_job_id_set = update_running_job_set(job_name, running_job_id_set)


def get_serving_and_standby_index_name(region, intention_index_status):
    ssm = boto3.client('ssm', region)

    response = ssm.get_parameter(
        Name=intention_index_status,
        WithDecryption=False
    )
    status = response['Parameter']['Value']

    # First Item is serving_index_name, Second Item is standby_index_name
    serving_index_name, standby_index_name = status.split(',')

    return serving_index_name, standby_index_name

def swap_serving_index_name(region, intention_index_status):
    ssm = boto3.client('ssm', region)
    
    serving_index_name, standby_index_name = get_serving_and_standby_index_name(region, intention_index_status)

    overwrite_value = f"{standby_index_name},{serving_index_name}"

    response = ssm.put_parameter(
        Name=intention_index_status,
        Value=overwrite_value,
        Type='String',
        Overwrite=True
    )
    return response

# step1 
logger.info(f"[Step1] get serving and standby index name...")
serving_index_name, standby_index_name = get_serving_and_standby_index_name(region, ssm_key_for_index_status)
logger.info(f"standby_index_name - {standby_index_name}")
logger.info(f"serving_index_name - {serving_index_name}")

# step2
logger.info(f"[Step2] start deleting of {standby_index_name}...")
delete_intentions(region, bucket, f"{path_prefix}/deleteset/", aos_endpoint, emb_model_endpoint, standby_index_name)
logger.info(f"[Step2] Finish {standby_index_name}'s deletion")

# step3
logger.info(f"[Step3] start adding of {standby_index_name}...")
add_intentions(region, bucket, f"{path_prefix}/addset/", aos_endpoint, emb_model_endpoint, standby_index_name, concurrent_runs_quota, job_name, company, emb_batch_size)
logger.info(f"[Step3] Finish {standby_index_name}'s updating")

# step4
reponse = swap_serving_index_name(region, ssm_key_for_index_status)
logger.info(f"[Step4] reponse of swap action : {str(reponse)}...")

# step5
logger.info(f"[Step5] start deleting of {serving_index_name}...")
delete_intentions(region, bucket, f"{path_prefix}/deleteset/", aos_endpoint, emb_model_endpoint, serving_index_name)
logger.info(f"[Step5] Finish {serving_index_name}'s deletion")

# step6
logger.info(f"[Step6] start adding of {serving_index_name}...")
add_intentions(region, bucket, f"{path_prefix}/addset/", aos_endpoint, emb_model_endpoint, serving_index_name, concurrent_runs_quota, job_name, company, emb_batch_size)
logger.info(f"[Step6] Finish {serving_index_name}'s updating")

