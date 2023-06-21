import boto3
import time
import os
import time
import argparse
import itertools

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

def start_job(glue_client, job_name, key_path):
    print('start job for {}'.format(key_path))    
    response = glue.start_job_run(
        JobName=job_name,
        Arguments={
            '--additional-python-modules': 'pdfminer.six==20221105,gremlinpython==3.6.3,langchain==0.0.162,beautifulsoup4==4.12.2',
            '--object_key': key_path,
            '--REGION': 'us-west-2',
            '--AOS_ENDPOINT': 'vpc-domain66ac69e0-nc8rzhegbk2b-qrle2hshwnbenn7kh4lvhtbyea.us-west-2.es.amazonaws.com',
            '--bucket' : '106839800180-23-06-07-05-29-20-bucket',
            '--EMB_MODEL_ENDPOINT': 'st-paraphrase-mpnet-base-v2-2023-06-12-10-42-37-917-endpoint'
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
    parser.add_argument('--bucket', type=str, default='106839800180-23-06-07-05-29-20-bucket', help='output file')
    parser.add_argument('--aos_endpoint', type=str, default='vpc-domain66ac69e0-nc8rzhegbk2b-qrle2hshwnbenn7kh4lvhtbyea.us-west-2.es.amazonaws.com', help='Opensearch domain endpoint')
    parser.add_argument('--emb_model_endpoint', type=str, default='st-paraphrase-mpnet-base-v2-2023-06-12-10-42-37-917-endpoint', help='embedding model endpoint')
    parser.add_argument('--path_prefix', type=str, default='ai-content/batch/', help='file path prefix')
    parser.add_argument('--concurrent_runs_quota', type=int, default=30, help='quota of concurrent job runs')
    parser.add_argument('--job_name', type=str, default='chatbotfroms3toaosF98BA633-nBaFEyoUneqK', help='job name')
    args = parser.parse_args()
    
    region = args.region
    bucket = args.bucket
    aos_endpoint = args.aos_endpoint
    emb_model_endpoint = args.emb_model_endpoint
    path_prefix = args.path_prefix
    concurrent_runs_quota = args.concurrent_runs_quota - 10
    job_name = args.job_name
    
    running_job_id_set = set()
    
    file_generator = list_s3_objects(s3, bucket_name=bucket, prefix=path_prefix)
    batch_size = 5
    batches = batch_generator(file_generator, batch_size)

    for batch in batches:
        key_list_str = ','.join(batch)
        while len(running_job_id_set) >= concurrent_runs_quota:
            print('wait for previous running job done')
            time.sleep(5)
            running_job_id_set = update_running_job_set(job_name, running_job_id_set)
            print('concurrent_running: {}'.format(running_job_id_set))
            
        running_job_id=start_job(glue, job_name, key_list_str)
        running_job_id_set.add(running_job_id)
        time.sleep(1)