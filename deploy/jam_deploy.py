#!/usr/bin/env python
# coding: utf-8

import os
import time
import json
import sys

from huggingface_hub import snapshot_download
from pathlib import Path
import sagemaker
import boto3
from sagemaker import image_uris
from sagemaker.utils import name_from_base


embedding_model = sys.argv[1]
region = sys.argv[2]

local_model_path = Path("./bge-en-model")
local_model_path.mkdir(exist_ok=True)
model_name = "BAAI/bge-large-en"
commit_hash = "48db88180e8c78ac7def7d8e359f2af4f7728a27"
snapshot_download(repo_id=model_name, revision=commit_hash, cache_dir=local_model_path)

session = boto3.Session()
account_id = session.client('sts').get_caller_identity()['Account']

print(f"account_id: {account_id}, region: {region}")

s3_client = boto3.client("s3", region_name=region)
sm_client = boto3.client("sagemaker", region_name=region)
smr_client = boto3.client("sagemaker-runtime", region_name=region)

bucket = f"{account_id}-knowledge-bucket"

s3_model_prefix = "LLM-RAG/workshop/bge-en-model"  # folder where model checkpoint will go
model_snapshot_path = list(local_model_path.glob("**/snapshots/*"))[0]
s3_code_prefix = "LLM-RAG/workshop/bge-en-code"
print(f"s3_code_prefix: {s3_code_prefix}")
print(f"model_snapshot_path: {model_snapshot_path}")
print(f"embedding_model: {embedding_model}, bucket: {bucket}")

s3_model_target = f"s3://{bucket}/{s3_model_prefix}/"

os.system(f'aws s3 cp --recursive {model_snapshot_path} {s3_model_target}')

inference_image_uri = (
    f"763104351884.dkr.ecr.{region}.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
)
print(f"Image going to be used is ---- > {inference_image_uri}")

os.system('mkdir -p bge-en-code')

with open('bge-en-code/model.py', 'w') as model_file:
    script_content = 'from djl_python import Input, Output\nimport torch\nimport logging\nimport math\nimport os\nfrom FlagEmbedding import FlagModel\n\ndevice = torch.device(\'cuda:0\' if torch.cuda.is_available() else \'cpu\')\nprint(f\'--device={device}\')\n\ndef load_model(properties):\n    tensor_parallel = properties["tensor_parallel_degree"]\n    model_location = properties[\'model_dir\']\n    if "model_id" in properties:\n        model_location = properties[\'model_id\']\n    logging.info(f"Loading model in {model_location}")\n\n    model =  FlagModel(model_location)\n    \n    return model\n\nmodel = None\n\ndef handle(inputs: Input):\n    global model\n    if not model:\n        model = load_model(inputs.get_properties())\n\n    if inputs.is_empty():\n        return None\n    data = inputs.get_as_json()\n    \n    input_sentences = None\n    inputs = data["inputs"]\n    if isinstance(inputs, list):\n        input_sentences = inputs\n    else:\n        input_sentences =  [inputs]\n        \n    is_query = data["is_query"]\n    instruction = data["instruction"]\n    logging.info(f"inputs: {input_sentences}")\n    logging.info(f"is_query: {is_query}")\n    logging.info(f"instruction: {instruction}")\n    \n    if is_query and instruction:\n        input_sentences = [ instruction + sent for sent in input_sentences ]\n        \n    sentence_embeddings =  model.encode(input_sentences)\n        \n    result = {"sentence_embeddings": sentence_embeddings}\n    return Output().add_as_json(result)\n'
    model_file.write(script_content)

option_s3url = f"s3://{bucket}/{s3_model_prefix}/"

with open('bge-en-code/serving.properties', 'w') as prop_file:
    settings = f'engine=Python\noption.tensor_parallel_degree=1\noption.s3url = {s3_model_target}\n'
    prop_file.write(settings)

with open('bge-en-code/requirements.txt', 'w') as req_file:
    settings = 'transformers==4.28.1\nFlagEmbedding\n'
    req_file.write(settings)

s3_code_artifact = f's3://{bucket}/{s3_code_prefix}/s2e_model.tar.gz'
upload_cmd = f'aws s3 cp s2e_model.tar.gz {s3_code_artifact}'
os.system('tar czvf s2e_model.tar.gz bge-en-code')
os.system(upload_cmd)
time.sleep(5)
print(upload_cmd)

model_name = embedding_model
execute_role_arn= f"arn:aws:iam::{account_id}:role/sagemaker_execute_role"

create_model_response = sm_client.create_model(
    ModelName=model_name,
    ExecutionRoleArn=execute_role_arn,
    PrimaryContainer={
        "Image": inference_image_uri,
        "ModelDataUrl": s3_code_artifact
    },
    
)
model_arn = create_model_response["ModelArn"]
print(f"Created Model: {model_arn}")

endpoint_config_name = f"{model_name}-config"
endpoint_name = f"{model_name}-endpoint"

endpoint_config_response = sm_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            "VariantName": "variant1",
            "ModelName": model_name,
            "InstanceType": "ml.g4dn.xlarge",
            "InitialInstanceCount": 1,
            "ContainerStartupHealthCheckTimeoutInSeconds": 15*60,
        },
    ],
)

create_endpoint_response = sm_client.create_endpoint(
    EndpointName=f"{endpoint_name}", EndpointConfigName=endpoint_config_name
)
print(f"Created Endpoint: {create_endpoint_response['EndpointArn']}")


resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
status = resp["EndpointStatus"]
print("Status: " + status)

while status == "Creating":
    time.sleep(60)
    resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
    status = resp["EndpointStatus"]
    print("Status: " + status)

