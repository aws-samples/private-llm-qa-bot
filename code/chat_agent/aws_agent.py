import json
import os
import logging
import re

from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain import PromptTemplate, SagemakerEndpoint
from typing import Any, Dict, List, Union,Mapping, Optional, TypeVar, Union
from langchain.chains import LLMChain
from langchain.llms.bedrock import Bedrock
from botocore.exceptions import ClientError
import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

credentials = boto3.Session().get_credentials()

class APIException(Exception):
    def __init__(self, message, code: str = None):
        if code:
            super().__init__("[{}] {}".format(code, message))
        else:
            super().__init__(message)


def handle_error(func):
    """Decorator for exception handling"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except APIException as e:
            logger.exception(e)
            raise e
        except Exception as e:
            logger.exception(e)
            raise RuntimeError(
                "Unknown exception, please check Lambda log for more details"
            )

    return wrapper

class llmContentHandler(LLMContentHandler):
    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({'inputs': prompt,'history':[],**model_kwargs})
        return input_str.encode('utf-8')
    
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["outputs"]


def service_org(query, llm):
    context = """{"leadership_team": {"Chen Erik": {"job_title": "Head of SSO"}, "Yang Zhang": {"job_title": "Sr. Mgr, AI/ML"}, "Troy Cui": {"job_title": "Sr. Mgr, Analytics GTM"}, "Jack Han": {"job_title": "Sr Manager, Database GTM"}, "Scott Zhou": {"job_title": "Sr. Mgr, Compute& Storage GTM"}, "Ping Wang": {"job_title": "Manager, GCR Compute EC2"}, "Ang Wang": {"job_title": "Manager, GCR SSO Storage"}, "John Bai": {"job_title": "Sr. Mgr, Security GTM"}, "Xavier Wang": {"job_title": "Sr Mgr, GCR Data Services SSA"}, "Peng Sun": {"job_title": "Sr Manager, CSM, Beijing"}, "Yunfei Lv": {"job_title": "Sr. Mgr, Network & Edge GTM"}, "Young Yang": {"job_title": "Sr Manager, HKT SSA"}, "Zhanling Chen": {"job_title": "Manager, GCR SSO AppMod"}, "Brenda Li": {"job_title": "HK SSO Lead"}, "Cathy Lai": {"job_title": "TWN SSO Lead"}, "Yu Guan": {"job_title": "Manager, ServiceLaunch"}}, "AI_ML": {"service_name": "AI_ML", "has_second_service": false, "specific_data": {"AI_ML": {"is_second_service": false, "service_name": "", "scope": {"SageMaker": {"GTMS": {"Yinuo He": ""}}, "NLP/LLM(Comprehend/Transcribe/Translate/Kendra/Polly/Lex)": {"GTMS": {"Jun Deng": ""}}, "Rekognition/Textract": {"PM": {"Yvonne Yang": ""}}, "CV+Program": {"PM": {"Yvonne Yang": ""}}, "North": {"SS": {"Yun Li": ""}, "SSA": {"Shishuai Wang": "", "Yuhui Liang": ""}}, "East - Named Account": {"SS": {"Bruce Liu": ""}, "SSA": {"Xueqing Li": ""}}, "East": {"SS": {"Jasmine Chen": ""}, "SSA": {"Hao Zheng": ""}}, "West": {"SS": {"Nick Jiang": ""}, "SSA": {"Qingyuan Tang": ""}}, "South": {"SS": {"Xin Hao": ""}, "SSA": {"Shaoyong Shen": ""}}, "XC": {"SSA": {"Zheng Zhang": ""}}, "Analytics": {"SSA": {"Yuanbo Li": ""}}}}}}, "Analytics": {"service_name": "Analytics", "has_second_service": false, "specific_data": {"Analytics": {"is_second_service": false, "service_name": "", "scope": {"South": {"SSA": {"Huang Xiao": "", "Pan Chao": ""}, "SS": {"Zhongfu Jing": ""}}, "East": {"SSA": {"Dalei Xu": ""}, "SS": {"Gracy Xu": ""}}, "North": {"SSA": {"Shijian Tang": "", "Pan Chao": ""}, "SS": {"Wuguang Zhu": ""}}, "FSI DNB": {"SSA": {"Neo Sun": ""}, "SS": {"Yuky Zhang": ""}}, "Redshift": {"SSA": {"Wenhui Yang": ""}, "GTMS": {"Dan Jin": ""}}, "West": {"SSA": {"Qingyuan Tang": ""}, "SS": {"Wuguang Zhu": ""}}, "AI/ML": {"SSA": {"Qingyuan Tang": "", "Yuanbo Li": ""}}, "ANA": {"SSA": {"Qingyuan Tang": "", "Yuanbo Li": ""}}, "G1K": {"SSA": {"Laurence Geng": ""}}, "G1K Auto": {"SS": {"Zhongfu Jing": ""}}, "CS auto": {"SS": {"Zhongfu Jing": ""}}, "Geo Central": {"SS": {"Zhongfu Jing": ""}}, "Fu Jian": {"SS": {"Zhongfu Jing": ""}}, "G1K exl Auto": {"SS": {"Gracy Xu": ""}}, "Shan Dong": {"SS": {"Wuguang Zhu": ""}}, "SMB": {"SS": {"Wuguang Zhu": ""}}, "NWCD": {"SS": {"Wuguang Zhu": ""}}, "Amazon OpenSearch": {"GTMS": {"Jason Li": ""}}, "EMR": {"GTMS": {"Goden Yao": ""}}, "Glue": {"GTMS": {"Goden Yao": ""}}, "Athena": {"GTMS": {"Goden Yao": ""}}, "Lake Formation": {"GTMS": {"Goden Yao": ""}}, "BI & Data Governance": {"GTMS": {"Jin Shen": ""}}, "Partner": {"GTMS": {"Jin Shen": ""}}, "Cross-Sell(AI/ML+ANA)": {"GTMS": {"Na Zhang": ""}}, "MSK/Kinesis": {"GTMS": {"Na Zhang": ""}}, "QuickSignt": {"GTMS": {"Na Zhang": ""}}}}}}, "AppMod": {"service_name": "AppMod", "has_second_service": false, "specific_data": {"AppMod": {"is_second_service": false, "service_name": "", "scope": {"Code/EKS-A/ECS-A/ECR": {"GTMS": {"Zhanling Chen": ""}, "SSA": {"Walkley He": "", "Pengyong Ye": ""}}, "Others(ROSA/Serverless App Repo/CloudMap/AppMesh/AppRunner)": {"GTMS": {"Zhanling Chen": ""}}, "Lambda/API GW": {"GTMS": {"Ping Ma": ""}, "SSA": {"Harold Sun": ""}}, "ECS/Fargate": {"GTMS": {"Han Li": ""}, "SSA": {"Dongdong Yang": ""}}, "Partner": {"GTMS": {"Ka Chen": ""}, "SSA": {"Bingjiao Yu": ""}}, "EKS": {"GTMS": {"Xing Wang": ""}, "SSA": {"Xiangyan Wang": ""}}, "MQ/SQS/SNS/ Step Functions/ EventBridge/MWAA": {"GTMS": {"Li Gong": ""}, "SSA": {"Elon Niu": ""}}, "Marketing": {"GTMS": {"Li Gong": ""}, "SSA": {"Elon Niu": ""}}}}}}, "CICE": {"service_name": "CICE", "has_second_service": true, "specific_data": {"Cloud_Intelligence": {"is_second_service": true, "service_name": "Cloud_Intelligence", "scope": {"AliCloud Competition": {"SS": {"Ethan Xie": ""}}, "Tencent Cloud Competition": {"SS": {"Ethan Xie": ""}}, "Azure Competition": {"SS": {"Ethan Xie": "", "Vivienne Xu": ""}}, "GCP Competition": {"SS": {"Ethan Xie": "", "Vivienne Xu": ""}}, "OCI Competition": {"SS": {"Ethan Xie": ""}}, "Field Enablement": {"SS": {"Ethan Xie": ""}}}}, "Cloud_Economics": {"is_second_service": true, "service_name": "Cloud_Economics", "scope": {"Business Value (BV)": {"SS": {"Nathan Mao": "", "Dave Sun": ""}, "SSA": {"Jiang Qi": ""}}, "Cloud Financial Management (CFM)": {"SS": {"Nathan Mao": "", "Dave Sun": ""}, "SSA": {"Jiang Qi": ""}}, "FinHack": {"SS": {"Nathan Mao": "", "Dave Sun": ""}, "SSA": {"Jiang Qi": ""}}}}}}, "Compute": {"service_name": "Compute", "has_second_service": false, "specific_data": {"Compute": {"is_second_service": false, "service_name": "", "scope": {"Savings Plans": {"GTMS": {"Ping Wang": ""}}, "C/R/M/T(Intel)": {"GTMS": {"Xu Cao": ""}, "SSA": {"Lei Wang": ""}}, "Graviton (C/R/M/T/X)": {"GTMS": {"Xu Cao": ""}, "SSA": {"Vincent Wang": ""}}, "X/U": {"GTMS": {"Xu Cao": ""}}, "Mac": {"GTMS": {"Xu Cao": ""}, "SSA": {"Lei Wang": ""}}, "AL": {"GTMS": {"Xu Cao": ""}, "SSA": {"Lei Wang": ""}}, "C/R/M/T(AMD)": {"GTMS": {"Kai Yao": ""}, "SSA": {"Quan Yuan": ""}}, "Graviton (I)": {"GTMS": {"Kai Yao": ""}, "SSA": {"Vincent Wang": ""}}, "I/D/H": {"GTMS": {"Kai Yao": ""}, "SSA": {"Jieling Ding": ""}}, "Lightsail": {"GTMS": {"Kai Yao": ""}, "SSA": {"Quan Yuan": ""}}, "Flexible Compute": {"GTMS": {"Kai Yao": ""}, "SSA": {"Jieling Ding": ""}}, "Graviton (c6gn, c7gn)": {"GTMS": {"Troy Liang": ""}, "SSA": {"Vincent Wang": ""}}, "EC2 Networking": {"GTMS": {"Troy Liang": ""}}}}}}, "Connect_SES_Pinpoint": {"service_name": "Connect_SES_Pinpoint", "has_second_service": false, "specific_data": {"Connect_SES_Pinpoint": {"is_second_service": false, "service_name": "", "scope": {"Connect": {"GTMS": {"Jinyi Li": ""}, "SSA": {"Lei Kang": ""}}, "SES": {"GTMS": {"Jinyi Li": ""}, "SSA": {"Lei Kang": ""}}, "Pinpoint": {"GTMS": {"Jinyi Li": ""}, "SSA": {"Lei Kang": ""}}}}}}, "Database": {"service_name": "Database", "has_second_service": false, "specific_data": {"Database": {"is_second_service": false, "service_name": "", "scope": {"RDS open source": {"GTMS": {"Jingyu Zhang": ""}}, "Aurora": {"GTMS": {"Jingyu Zhang": ""}}, "RDS SQL Server": {"GTMS": {"Yujia Wang": ""}}, "Oracle": {"GTMS": {"Yujia Wang": ""}, "SSA": {"Chandler Lv": "", "Xiaohua Tang": ""}}, "Babelfish": {"GTMS": {"Yujia Wang": ""}}, "ElastiCache": {"GTMS": {"Michael Dai": ""}, "SSA": {"Chandler Lv": "", "Lili Ma": "", "Chuan Jin": "", "Yang Chen": ""}}, "MemoryDB": {"GTMS": {"Michael Dai": ""}, "SSA": {"Lili Ma": ""}}, "DocumentDB": {"GTMS": {"Zaijun An": ""}, "SSA": {"Bingbing Liu": ""}}, "Timestream": {"GTMS": {"Zaijun An": ""}, "SSA": {"Chandler Lv": "", "Bingbing Liu": ""}}, "DynamoDB": {"GTMS": {"Fred Yu": ""}, "SSA": {"Chandler Lv": "", "Amy Li": ""}}, "Keyspaces": {"GTMS": {"Fred Yu": ""}}, "DMS": {"GTMS": {"Fred Yu": ""}, "SSA": {"Chandler Lv": ""}}, "Others": {"GTMS": {"Fred Yu": ""}}, "DNB North": {"SS": {"Min Xu": ""}}, "SMB Startups": {"SS": {"Min Xu": ""}}, "DNB East": {"SS": {"Neos Huang": ""}}, "ENT Auto": {"SS": {"Neos Huang": ""}}, "G1k Retail": {"SS": {"Neos Huang": ""}}, "GEO Central": {"SS": {"Neos Huang": ""}}, "Aurora MySQL": {"SSA": {"Chandler Lv": "", "Bingbing Liu": "", "Lili Ma": "", "Amy Li": "", "Chuan Jin": "", "Yang Chen": ""}}, "Neptune": {"SSA": {"Chandler Lv": ""}}, "Aurora PostgreSQL": {"SSA": {"Xiaohua Tang": ""}}}}}}, "HPC": {"service_name": "HPC", "has_second_service": false, "specific_data": {"HPC": {"is_second_service": false, "service_name": "", "scope": {"HPC Hybrid": {"GTMS": {"Qing Wan": ""}, "SSA": {"Bing Liu": ""}}, "East Partner": {"GTMS": {"Yu Geng": ""}}, "HPC instance (hpc6a/hpc6id, x2iezn, EFA)": {"GTMS": {"Yu Geng": ""}, "SSA": {"Bing Liu": ""}}, "HPC software (SOCA, Batch, Parallel cluster, etc)": {"GTMS": {"Yu Geng": ""}, "SSA": {"Bing Liu": ""}}, "HPC+": {"GTMS": {"Yu Geng": ""}, "SSA": {"Bing Liu": ""}}, "East HCLS": {"GTMS": {"Troy Liang": ""}}}}}}, "Hybrid": {"service_name": "Hybrid", "has_second_service": false, "specific_data": {"Hybrid": {"is_second_service": false, "service_name": "", "scope": {"Outposts Rack/ Outposts Server": {"GTMS": {"Qing Wan": ""}, "SSA": {"Larry Yang": ""}}, "Local Zone": {"GTMS": {"Qing Wan": ""}, "SSA": {"Larry Yang": ""}}, "Wavelength": {"GTMS": {"Qing Wan": ""}, "SSA": {"Larry Yang": ""}}}}}}, "IoT": {"service_name": "IoT", "has_second_service": false, "specific_data": {"IoT": {"is_second_service": false, "service_name": "", "scope": {"CN East": {"GTMS": {"Karen Fan": ""}}, "HKT": {"GTMS": {"Anderson Hsiao": ""}}, "KVS": {"GTMS": {"Xinggao Xia": ""}, "SSA": {"Zihang Huang": ""}}, "CN North": {"GTMS": {"Xinggao Xia": ""}}, "CN Mainland": {"GTMS": {"Yucheng Tsai": ""}}, "CN South": {"GTMS": {"Yucheng Tsai": ""}}, "IoT": {"SSA": {"Zihang Huang": ""}}}}}}, "Network_Edge": {"service_name": "Network_Edge", "has_second_service": true, "specific_data": {"Networking": {"is_second_service": true, "service_name": "Networking", "scope": {"Connectivity Service": {"GTMS": {"CuiCui Cui": "", "Kevin Liu": ""}}, "DX": {"GTMS": {"Harvey Yang": "", "Jingqing Xu": ""}}, "Data Transfer(DTO via Internet/Static BGP, DTAZ, DTIR)": {"GTMS": {"Henry Peng": ""}}, "Program": {"GTMS": {"Hongyan Zhao": ""}}, "Networking": {"SSA": {"Yibai Liu": ""}}}}, "Edge": {"is_second_service": true, "service_name": "Edge", "scope": {"DNB": {"SS": {"Bo Li": ""}}, "FSI": {"SS": {"Bo Li": ""}}, "HKT": {"SS": {"Leo Cheng": ""}}, "ENT": {"SS": {"Shuhai Liu": ""}}, "MMT": {"SS": {"Shuhai Liu": ""}}, "Key Account": {"SS": {"Yufei Lv": ""}}, "Edge Others": {"SSA": {"Sam Ye": ""}}, "Edge East": {"SSA": {"Alex Cui": ""}}, "Edge South": {"SSA": {"Cheng Chen": ""}}, "Edge North": {"SSA": {"Jason Wang": ""}}}}}}, "Security": {"service_name": "Security", "has_second_service": false, "specific_data": {"Security": {"is_second_service": false, "service_name": "", "scope": {"ENT": {"GTMS": {"Jason Jiang": ""}, "SSA": {"Di Zhao": ""}}, "ESS": {"GTMS": {"Jason Jiang": ""}, "SSA": {"Xudong Wang": ""}}, "Data Crypto(PCA/CA/CloudHSM/Macie/KMS/SecretesManager)": {"GTMS": {"Jason Jiang": ""}, "SSA": {"Di Zhao": ""}}, "Mgmt Tools(Config/CloudTrail/CloudWatch/CWE/CWL/ControlTower)": {"GTMS": {"Jason Jiang": ""}, "SSA": {"Yang Li": ""}}, "AuditManger/Detective/GuardDuty/Inspector/SecurityHUB": {"GTMS": {"Jason Jiang": ""}}, "G1K": {"GTMS": {"Ying Zhou": ""}, "SSA": {"Yang Li": ""}}, "Compliance": {"GTMS": {"Ying Zhou": ""}}, "SMB": {"GTMS": {"Jiahui Chen": ""}, "SSA": {"Xudong Wang": ""}}, "Identity(Cognito/AD Connector/Managed AD/IAM)": {"GTMS": {"Jiahui Chen": ""}, "SSA": {"Matt Yu": ""}}, "Cloud Directory": {"GTMS": {"Jiahui Chen": ""}}, "Single Sign-On": {"GTMS": {"Jiahui Chen": ""}}, "DNB": {"GTMS": {"Xiaojiang Wen": ""}, "SSA": {"Matt Yu": ""}}, "FSI DNB": {"GTMS": {"Xiaojiang Wen": ""}, "SSA": {"Matt Yu": ""}}, "Perimeter Protection(Shield Advance/FirewallManager/NetworkFirewall/WAF)": {"GTMS": {"Xiaojiang Wen": ""}, "SSA": {"Matt Yu": ""}}}}}}, "Storage": {"service_name": "Storage", "has_second_service": true, "specific_data": {"By_Services": {"is_second_service": true, "service_name": "By_Services", "scope": {"S3 & Glacier": {"GTMS": {"Ryan Lian": ""}, "SSA": {"Yiyang Dai": ""}}, "SnowFamily": {"GTMS": {"Ryan Lian": ""}, "SSA": {"Zhixuan Li": ""}}, "Storage GW": {"GTMS": {"Ryan Lian": ""}, "SSA": {"Zhixuan Li": ""}}, "Datasync": {"GTMS": {"Ryan Lian": ""}, "SSA": {"Zhixuan Li": ""}}, "AWS Transfer Family": {"GTMS": {"Ryan Lian": ""}, "SSA": {"Zhixuan Li": ""}}, "AWS Backup": {"GTMS": {"Ryan Lian": ""}, "SSA": {"Zhixuan Li": ""}}, "CloudEndure/DRS": {"GTMS": {"Ryan Lian": ""}, "SSA": {"Zhixuan Li": ""}}, "EFS": {"GTMS": {"Yanyun Chen": ""}, "SSA": {"Zhida Wang": ""}}, "FSx Lustre": {"GTMS": {"Yanyun Chen": ""}, "SSA": {"Dawei Wang": ""}}, "FSx ZFS": {"GTMS": {"Yanyun Chen": ""}, "SSA": {"Dawei Wang": ""}}, "FSx Windows": {"GTMS": {"Yanyun Chen": ""}, "SSA": {"Dawei Wang": ""}}, "FSx OnTap": {"GTMS": {"Yanyun Chen": ""}, "SSA": {"Dawei Wang": ""}}, "File Cache": {"GTMS": {"Yanyun Chen": ""}, "SSA": {"Dawei Wang": ""}}, "EBS": {"SSA": {"Allen Xie": ""}}}}, "By_Account_teams": {"is_second_service": true, "service_name": "By_Account_teams", "scope": {"G1K & Telecom": {"GTMS": {"Ang Wang": ""}, "SS": {"Lijun Fan": ""}}, "SMB Standard": {"GTMS": {"Ang Wang": ""}, "SS": {"Muni Zhang": ""}}, "CS DNB": {"GTMS": {"Ryan Lian": ""}}, "CS ENT-GEO/East": {"GTMS": {"Yanyun Chen": ""}, "SS": {"Lijun Fan": ""}}, "CS ENT- FSI DNB/MOB/SMV/South": {"GTMS": {"Yanyun Chen": ""}}, "CS ENT-Others": {"GTMS": {"Yanyun Chen": ""}, "SS": {"Muni Zhang": ""}}, "PS": {"GTMS": {"Yanyun Chen": ""}, "SS": {"Lijun Fan": ""}}}}}}, "Service Launch": {"service_name": "Service Launch", "has_second_service": false, "specific_data": {"Service Launch": {"is_second_service": false, "service_name": "", "scope": {"Networking": {"TPM": {"Fuyin Wang": ""}}, "Edge": {"TPM": {"Fuyin Wang": ""}}, "IPv6 related features": {"TPM": {"Fuyin Wang": ""}}, "Document Improvement": {"TPM": {"Grant Wang": ""}}, "Security": {"TPM": {"Neo Xiang": ""}}, "Mgmt Tool": {"TPM": {"Neo Xiang": ""}}, "Customer Support Issue Improvement": {"TPM": {"Reinhard Xu": ""}}, "Saving Plans": {"TPM": {"River Xie": ""}}, "FOOB": {"TPM": {"River Xie": ""}}, "Compute": {"TPM": {"Winson Tam": ""}}, "Storage": {"TPM": {"Winson Tam": ""}}, "Pricing Calculator": {"TPM": {"Tony Jin": ""}}, "Database": {"TPM": {"Cynthia Chen": ""}}, "Analytics": {"TPM": {"Cynthia Chen": ""}}}}}}}"""
    
    prompt_tmp = """
        你是云服务AWS的智能客服机器人AWSBot

        给你 SSO (Service Specialist Organization) 的组织信息
        {context}

        Job role (角色, 岗位类型) description:
        - GTMS: Go To Market Specialist
        - SS: Specialist Sales
        - SSA: Specialist Solution Architechure
        - TPM: 
        - PM: Project Manager

        Scope means job scope
        service_name equal to business unit

        If the context does not contain the knowleage for the question, truthfully says you does not know.
        Don't put two people's names together. For example, zheng zhang not equal to zheng hao and xueqing not equal to Xueqing Lai

        Find out the most relevant context, and give the answer according to the context
        Skip the preamble; go straight to the point.
        Only give the final answer.
        Do not repeat similar answer.
        使用中文回复，人名不需要按照中文习惯回复

        {question}
        """

    def create_prompt_templete(prompt_template):
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context",'question','chat_history']
        )
        return PROMPT
    prompt = create_prompt_templete(prompt_tmp) 
    llmchain = LLMChain(llm=llm,verbose=False,prompt = prompt)
    answer = llmchain.run({'question':query, "context": context})
    logger.info(f'context length: {len(context)}, prompt {prompt}')
    answer = answer.strip()
    return answer

@handle_error
def lambda_handler(event, context):
    params = event.get('params')
    param_dict = params
    query = param_dict["query"]
    intention = param_dict["intention"]     
    
    
    use_bedrock = event.get('use_bedrock')
    
    region = os.environ.get('region')
    llm_model_endpoint = os.environ.get('llm_model_endpoint')
    llm_model_name = event.get('llm_model_name', None)
    logger.info("region:{}".format(region))
    logger.info("params:{}".format(params))
    logger.info("llm_model_name:{}, use_bedrock: {}".format(llm_model_name, use_bedrock))
    logger.info("llm_model_endpoint:{}".format(llm_model_endpoint))

    parameters = {
        "temperature": 0.01,
    }

    llm = None
    if not use_bedrock:
        logger.info(f'not use bedrock, use {llm_model_endpoint}')
        llmcontent_handler = llmContentHandler()
        llm=SagemakerEndpoint(
                endpoint_name=llm_model_endpoint, 
                region_name=region, 
                model_kwargs={'parameters':parameters},
                content_handler=llmcontent_handler
            )
    else:
        boto3_bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name=region
        )
    
        parameters = {
            "max_tokens_to_sample": 8096,
            "stop_sequences": ["\nObservation"],
            "temperature":0.01,
            "top_p":0.85
        }
        
        model_id ="anthropic.claude-instant-v1" if llm_model_name == 'claude-instant' else "anthropic.claude-v2"
        llm = Bedrock(model_id=model_id, client=boto3_bedrock, model_kwargs=parameters)

    if intention == "Service角色查询":
        answer = service_org(query, llm)
    else:
        return "抱歉，service 差异查询功能还在开发中，暂时无法回答"
    
    log_dict = {"answer" : answer , "question": query }
    log_dict_str = json.dumps(log_dict, ensure_ascii=False)
    logger.info(log_dict_str)
    pattern = r'^根据[^，,]*[,|，]'
    answer = re.sub(pattern, "", answer)
    logger.info(answer)
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body':answer
    }