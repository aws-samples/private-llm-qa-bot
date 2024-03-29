import boto3
import time
import os
import time
import argparse
import itertools
import datetime
import math

glue = boto3.client('glue')
s3 = boto3.client('s3')

def update_running_job_set(job_name, running_job_id_set):
    response = glue.get_job_runs(JobName=job_name)

    # Filter the finished job runs
    finish_runs_id = [run['Id'] for run in response['JobRuns'] if run['JobRunState'] in ['STOPPED', 'SUCCEEDED', 'FAILED', 'ERROR', 'TIMEOUT'] ]
    
    return running_job_id_set - set(finish_runs_id)
        
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
def count_s3_files(s3_client, bucket_name, prefix):
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    file_count = 0
    for page in page_iterator:
        for obj in page.get('Contents', []):
            if not obj['Key'].endswith('/'):
                file_count += 1
    return file_count


def start_job(glue_client, job_name, key_path, aos_endpoint, emb_model_endpoint, bucket, region_name, publish_date, company, emb_batch_size=5):
    print('start job for {} at {}'.format(key_path, str(publish_date)))   
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
            '--emb_batch_size' : str(emb_batch_size)
            })  
    return response['JobRunId']

def batch_generator(generator, batch_size):
    while True:
        batch = list(itertools.islice(generator, batch_size))
        if not batch:
            break
        yield batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', type=str, default='us-west-2', help='region_name')
    parser.add_argument('--bucket', type=str, default='106839800180-23-07-17-15-02-02-bucket', help='output file')
    parser.add_argument('--aos_endpoint', type=str, default='vpc-domainc6g17c529-y8fodglwythm-5he24vsymfmfh5derkzrj4l4ry.us-west-2.es.amazonaws.com', help='Opensearch domain endpoint')
    parser.add_argument('--emb_model_endpoint', type=str, default='st-paraphrase-mpnet-node10-2023-07-30-16-39-58-975-endpoint', help='embedding model endpoint')
    parser.add_argument('--path_prefix', type=str, default='ai-content/batch/', help='file path prefix')
    parser.add_argument('--concurrent_runs_quota', type=int, default=50, help='quota of concurrent job runs')
    parser.add_argument('--job_name', type=str, default='chatbotfroms3toaosF98BA633-QxSQwoaGE1K9', help='job name')
    parser.add_argument('--company', type=str, default='default', help='tenant name')
    parser.add_argument('--emb_batch_size', type=int, default=5, help='embedding batch inference size')
    args = parser.parse_args()
    
    region = args.region
    bucket = args.bucket
    aos_endpoint = args.aos_endpoint
    emb_model_endpoint = args.emb_model_endpoint
    path_prefix = args.path_prefix
    concurrent_runs_quota = args.concurrent_runs_quota
    job_name = args.job_name
    company = args.company
    emb_batch_size = args.emb_batch_size
    
    running_job_id_set = set()

    file_cnt = count_s3_files(s3, bucket_name=bucket, prefix=path_prefix)
    batch_size = math.ceil(file_cnt / concurrent_runs_quota)
    print(f"file_cnt: {file_cnt}, batch_size:{batch_size}")
    
    file_generator = list_s3_objects(s3, bucket_name=bucket, prefix=path_prefix)
    
    batches = batch_generator(file_generator, batch_size)
    publish_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") 

    for idx, batch in enumerate(batches):
        key_list_str = ','.join(batch)
        while len(running_job_id_set) >= concurrent_runs_quota:
            print('wait for previous running job done')
            time.sleep(100)
            running_job_id_set = update_running_job_set(job_name, running_job_id_set)
            print('concurrent_running: {}'.format(running_job_id_set))
            
        running_job_id=start_job(glue, job_name, key_list_str, aos_endpoint, emb_model_endpoint, bucket, region, publish_date, company, emb_batch_size)
        running_job_id_set.add(running_job_id)
        # sleep_seconds = len(running_job_id_set) * 2
        print("[{}] running job count: {}".format(idx, len(running_job_id_set)))
        time.sleep(15)