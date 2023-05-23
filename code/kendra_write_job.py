import boto3
from botocore.exceptions import ClientError
import pprint
import time

"""
加载 s3 文档进 kendra index 流程：
1.创建 s3 data source（如果当前没有，见function create_data_source())
2.上传文档到 s3 配置的目录（在创建 s3 data source 时配置的 bucket 与 prefix)
3.触发 sync s3 data source（见function sync_data_source())
"""

# test data source create
INDEX_ID = 'f36f5962-4ca8-4a65-9c60-5b813e5f46bc'
DATA_SOURCE_NAME = 's3-doc-source'
DATA_SOURCE_ROLE_ARN = 'arn:aws:iam::946277762357:role/service-role/AmazonKendra-s3-chat-doc-role'
DATA_SOURCE_S3_BUCKET_NAME = 'chatbot-llm-analytics'

def create_data_source(data_source_name, data_source_role_arn, s3_bucket_name, index_id, lang='zh', include_prefix=['docs']):
    """
    create a s3 data source, please only create if the data_source does not exists.
    :param data_source_name:
    :param data_source_role_arn: the role must have access to read s3_bucket
    :param s3_bucket_name: docs will be uploaded to a s3 bucket
    :param index_id: kendra index id
    :param lang: doc language, Chinese(zh) by default
    :param include_prefix: specify included prefixs, 'docs' by default
    :return: data_source_id
    """

    print("Create an S3 data source.")

    kendra = boto3.client("kendra", region_name="us-east-1")

    # Configure the data source
    configuration = {"S3Configuration":
        {
            "BucketName": s3_bucket_name,
            'InclusionPrefixes': include_prefix
        }
    }

    try:
        data_source_response = kendra.create_data_source(
            Name=data_source_name,
            RoleArn=data_source_role_arn,
            Type="S3",
            Configuration=configuration,
            IndexId=index_id,
            LanguageCode=lang
        )

        if data_source_response['ResponseMetadata']['HTTPStatusCode'] == 200:
            print("Created datasource ", data_source_name, " successfully.")

        return data_source_response['Id']

    except  ClientError as e:
        print("%s" % e)


def sync_data_source(data_source_id, index_id, sync_wait=0):
    """
    sync a data source
    :param data_source_id: data source id
    :param index_id: Index id
    :param sync_wait: whether to wait synchronously
    :return:
    """
    print("Synchronize the data source.")

    kendra = boto3.client("kendra", region_name="us-east-1")

    sync_response = kendra.start_data_source_sync_job(
        Id=data_source_id,
        IndexId=index_id
    )

    pprint.pprint(sync_response)

    print("Wait for the data source to sync with the index.")

    while sync_wait:

        jobs = kendra.list_data_source_sync_jobs(
            Id=data_source_id,
            IndexId=index_id
        )

        # For this example, there should be one job
        status = jobs["History"][0]["Status"]

        print(" Syncing data source. Status: " + status)
        if status != "SYNCING":
            break
        time.sleep(60)


# create a data source, only create if
#data_source_id = create_data_source(DATA_SOURCE_NAME, DATA_SOURCE_ROLE_ARN, DATA_SOURCE_S3_BUCKET_NAME, INDEX_ID)
#print(data_source_id)

# test data source sync
DATA_SOURCE_ID = '78efa31d-ce99-4ad7-9c1b-9a6d2121f0f3'
sync_data_source(DATA_SOURCE_ID, INDEX_ID, 1)
