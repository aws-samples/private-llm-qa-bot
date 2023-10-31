import json
import os
import logging

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

context = """"""

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


@handle_error
def lambda_handler(event, context):
    region = os.environ.get('region')
    llm_model_endpoint = os.environ.get('llm_model_endpoint')
    llm_model_name = event.get('llm_model_name', None)
    params = event.get('params')
    use_bedrock = event.get('use_bedrock')
    role_a = event.get('role_a', 'user')
    role_b = event.get('role_a', 'bot')
    
    logger.info("region:{}".format(region))
    logger.info("params:{}".format(params))
    logger.info("llm_model_name:{}".format(llm_model_name))
    logger.info("llm_model_endpoint:{}".format(llm_model_endpoint))

    param_dict = params
    query = param_dict["query"]
    history = param_dict["history"]

    history_with_role = [ "{}: {}".format(role_a if idx % 2 == 0 else role_b, item) for idx, item in enumerate(history) ]
    history_str = "\n".join(history_with_role)

    parameters = {
        "temperature": 0.01,
    }

    llm = None
    if not use_bedrock:
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
            "max_tokens_to_sample": 20,
            "stop_sequences": ["\n\n"],
            "temperature":0.01,
            "top_p":0.85
        }
        
        model_id ="anthropic.claude-instant-v1" if llm_model_name == 'claude-instant' else "anthropic.claude-v2"
        llm = Bedrock(model_id=model_id, client=boto3_bedrock, model_kwargs=parameters)
    
    B_Role="AWSBot"
    SYSTEM_ROLE_PROMPT = '你是云服务AWS的智能客服机器人AWSBot'
    promtp_tmp = """
    {system_role_prompt} {role_bot}

    {context}
    roles（角色） description:
    - GTMS: Go To Market Specialist
    - SS: Specialist Sales
    - SSA: Specialist Solution Architechure
    - TPM: 
    - PM: Project Manager

    If the context does not contain the knowleage for the question, truthfully says you does not know.
    if the parent node key has overlap with sibling child node, ask question to conform the parent node.
    if ask for the person's responsibility, must give all services or scope which in charged by them.
    Skip the preamble; go straight to the point.

    使用中文回复，人名不需要按照中文习惯回复

    {question}"""

    def create_prompt_templete(prompt_template):
        PROMPT = PromptTemplate(
            template=prompt_template,
            partial_variables={'system_role_prompt':SYSTEM_ROLE_PROMPT},
            input_variables=["context",'question','chat_history','role_bot']
        )
        return PROMPT
    prompt = create_prompt_templete(promtp_tmp) 
    llmchain = LLMChain(llm=llm,verbose=False,prompt = prompt)
    # query_input = "SSO 的负责人是谁"

    answer = llmchain.run({'question':query, 'role_bot':B_Role, "context": context})
    # print(answer)
    # llmchain = LLMChain(llm=llm, verbose=False, prompt=prompt_template)
    # answer = llmchain.run({'history':history_str, "cur_query":query})
    answer = answer.strip()

    log_dict = { "history" : history, "answer" : answer , "question": query }
    log_dict_str = json.dumps(log_dict, ensure_ascii=False)
    logger.info(log_dict_str)
    return answer.strip('"')
