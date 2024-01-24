import json
from typing import Any, Dict, List, Union,Mapping, Optional, TypeVar, Union
import os
import boto3
import requests
from pydantic import BaseModel,ValidationInfo, field_validator, Field,ValidationError
import re



class AWSSevicePriceRequest(BaseModel):
    region_name: str
    service_code: str

    @classmethod
    def validate_ec2_instance_type(cls,instance_type):
        # support other instance ml.m5.xlarge
        # pattern = r'^(?:[a-z0-9][a-z0-9.-]*[a-z0-9])?(?:[a-z](?:[a-z0-9-]*[a-z0-9])?)?(\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)*\.[a-z0-9]{2,63}$'
        
        ## only ec2, for m5.xlarge
        pattern = r"^([a-z0-9]+\.[a-z0-9]+)$"

        return re.match(pattern, instance_type) is not None and not instance_type.endswith(".")
    
    @classmethod
    def validate_region_name(cls,region_name):
        pattern = r"^[a-z]{2}(-gov)?-(central|east|north|south|west|northeast|northwest|southeast|southwest)-\d$"
        return re.match(pattern, region_name) is not None

    @field_validator('region_name')
    def validate_region(cls, value:str,info: ValidationInfo):
        if not cls.validate_region_name(value):
            # return f'value must be one of {allowed_values}'
            raise ValueError(f"{value} is not a valid AWS region name.")
        return value
    



def remote_proxy_call(**args):
    api = os.environ.get('api_endpoint')
    key = os.environ.get('api_key')
    payload = json.dumps(args)
    if not api or not key:
        return None
    try:
        resp = requests.post(api,headers={"Content-Type":"application/json","Authorization":f"Bearer {key}"},data=payload)
        data = resp.json()
        return data.get('message')
    except Exception as e:
        print(e)
        return None
    

def list_all_services():
    client = boto3.client('pricing',region_name='us-east-1')
    ret = []
    response = client.describe_services(
        FormatVersion='aws_v1',
        MaxResults = 100,
    )
    ret += response['Services']
    next_token = response['NextToken']
    while next_token:
        response = client.describe_services(
            FormatVersion='aws_v1',
            MaxResults = 100,
            NextToken = next_token
        )
        next_token = response.get('NextToken')
        ret += response['Services']
    return ret

def make_examples():
    import random
    region_alias_en = [('us west 2','us-west-2'),('japan','ap-northeast-1'),('us east 1','us-east-1'),('china','cn-north-1'),('ningxia','cn-northwest-1')]
    region_alias_zh = [('美西2','us-west-2'),('日本','ap-northeast-1'),('美东1','us-east-1'),('中国','cn-north-1'),('宁夏','cn-northwest-1')]
    all_services = list_all_services()
    service_codes = [s['ServiceCode'] for s in all_services ] 
    examples_en = []
    examples_zh = []
    for service_code in service_codes:
        region_en = random.choice(region_alias_en)
        region_zh = random.choice(region_alias_zh)
        examples_en.append({"query": f"what is the pricing of {service_code} in {region_en[0]}", "detection":{"func" : "aws_service_price", "param" : {"service_code":f"{service_code}" ,"region_name" : f"{region_en[1]}"}}})
        examples_zh.append({"query": f" {region_zh[0]} {service_code} 服务的定价是怎样的？ ", "detection":{"func" : "aws_service_price", "param" : {"service_code":f"{service_code}" ,"region_name" : f"{region_zh[1]}"}}})
    return examples_zh+examples_en
    

def parse_price(products):
    ret = []
    for product in products:
        product = json.loads(product)
        info = {'product':product['product'],"term":product['terms']}
        ret.append(json.dumps(info))
    return ret
                

def query_aws_service_price(**args) -> Union[str,None]:  
    request = AWSSevicePriceRequest(**args)
    region = request.region_name
    service_code = request.service_code
    if region.startswith('cn-'):
        return remote_proxy_call(**args)
    else:
        pricing_client = boto3.client('pricing',region_name='us-east-1')
        filters = [
            {
                    'Type': 'TERM_MATCH',
                    'Field': 'servicecode',
                    'Value': service_code,
            },
            {
                'Type': 'TERM_MATCH',
                'Field': 'regionCode',
                'Value': region
            }
        ]
        response = pricing_client.get_products(
            ServiceCode=service_code,
            Filters=filters
        )
        poducts = parse_price(response['PriceList'])
        return '\n'.join(poducts)


if __name__ == "__main__":
    # args = {'region':'us-east-1','term':'Reserved','purchase_option':'All Upfront'}
    print(query_aws_service_price(region_name='us-east-1',service_code='AmazonKinesisFirehose'))
    # print(make_examples())