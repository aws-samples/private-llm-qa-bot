import json
import os
import logging
import re
import argparse
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.prompts import PromptTemplate
from langchain.llms import SagemakerEndpoint
from typing import Any, Dict, List, Union,Mapping, Optional, TypeVar, Union
from langchain.chains import LLMChain
from langchain.llms.bedrock import Bedrock
from botocore.exceptions import ClientError
import boto3
import requests
from pydantic import BaseModel
from tools.get_price import query_ec2_price
from tools.service_org_demo import service_org

logger = logging.getLogger()
logger.setLevel(logging.INFO)

credentials = boto3.Session().get_credentials()
BEDROCK_REGION = None

REFUSE_ANSWER = '对不起, 根据{func_name}({args}),没有查询到您想要的信息，请您更具体的描述下您的要求.'

FUNCTION_CALL_TEMPLATE = """here is a list of functions you can use, contains in <tools> tags

<tools>
{functions}
</tools>

Given you a task, you need to :
1. Decide which tool need to be chosen to solve the task, if no tool is chosen, you should reply “I don't know”
2. Once you decide a tool, please extract the required parameters and function name from the task. 
3. Please response in json format such as {{"name":"function name", "arguments":{{"x":1,"y":1}}}}, and enclose the response in <function_call></function_call> tag, 

Task: {task}"""


API_SCHEMA = [
                {
                "name": "query_ec2_price",
                "description": "query the price of AWS ec2 instance",
                "parameters": {
                    "type": "object",
                    "properties": {
                         "instance_type": {
                            "type": "string",
                            "description": "the AWS ec2 instance type, for example, c5.xlarge, m5.large, t3.mirco, g4dn.2xlarge, if it is a partial of the instance type, you should try to auto complete it. for example, if it is r6g.2x, you can complete it as r6g.2xlarge",
                        },
                        "region": {
                            "type": "string",
                            "description": "the AWS region name where the ec2 is located in, for example us-east-1, us-west-1, if it is common words such as 'us east 1','美东1','美西2',you should try to normalize it to standard AWS region name, for example, 'us east 1' is normalized to 'us-east-1', '美东2' is normalized to 'us-east-2','美西2' is normalized to 'us-west-2','北京' is normalized to 'cn-north-1', '宁夏' is normalized to 'cn-northwest-1', '中国区' is normalized to 'cn-north-1'",
                        },
                        "os": {
                            "type": "string",
                            "description": "the operating system of ec2 instance,the valid value should be 'Linux' or 'Windows' ",
                        },
                        "term": {
                            "type": "string",
                            "description": "the payment term,the valid value should be 'OnDemand' or 'Reserved' ",
                        },
                        "purchase_option": {
                            "type": "string",
                            "description": "the purchase option of Reserved instance, the valid value should be 'No Upfront', 'Partial Upfront' or 'All Upfront' ",
                        },
                    },
                    "required": ["instance_type"],
                    },
                },
                {
                "name": "service_org",
                "description": "query the contact person in the organization",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "user's original input for query the service contact person",
                        },
                        "service": {
                            "type": "string",
                            "description": "the AWS service name ",
                        },
                    },
                    "required": ["query"],
                    },
                },
            ]


CONTEXT_TEMPLATE = """
        You are acting as an AWS assistant, please based on the context in <context></context>, answer user's question.
        Skip the preamble, go straight into the answer. if the context is empty,refuse to response politely.

        <context>
        {context}
        </context>

        Question: {question}
        """

class AgentTools(BaseModel):
    function_map: dict = {}
    api_schema: list
    llm: Any

    def register_tool(self,name:str,func:callable) -> None:
        self.function_map[name] = func
    
    def check_tool(self,name:str) -> bool:
        return name in self.function_map

    def _tool_call(self,name,**args) -> Union[str,None]:
        callback_func = self.function_map.get(name)
        return callback_func(**args) if callback_func else None

    @staticmethod
    def extract_function_call(content: str):
        pattern = r"<function_call>(.*?)</function_call>"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            function_call_content = match.group(1)
            try:
                function_call_dict = json.loads(function_call_content)
                function_call_dict['arguments'] = json.dumps(function_call_dict.get("arguments"),ensure_ascii=False)
                return function_call_dict
            except:
                return None
        else:
            return None


    def dispatch_function_call(self,query:str):
        ##parse the args
        prompt = PromptTemplate(
                template=FUNCTION_CALL_TEMPLATE,
                input_variables=["functions",'task']
            )
        llmchain = LLMChain(llm=self.llm,verbose=False,prompt = prompt)
        answer = llmchain.run({'functions':self.api_schema, "task": query})
        function_call = AgentTools.extract_function_call(answer)
        print(f"****use function_call****:{function_call}")
        if not function_call:
            return None,None,None
        try:
            args = json.loads(function_call['arguments'])
            func_name = function_call['name']
            result = self._tool_call(func_name,**args)
            return result,func_name,args
        except Exception as e:
            print(str(e))
            logger.info(str(e))
            return None,None,None


    def _add_context_answer(self,query,context) ->str:
        if not context:
            return None 
        prompt = PromptTemplate(
                template=CONTEXT_TEMPLATE,
                input_variables=["context",'question']
            )
        llmchain = LLMChain(llm=self.llm,verbose=False,prompt = prompt)

        logger.info(f'llm input:{CONTEXT_TEMPLATE.format(context=context,question=query)}')
        answer = llmchain.run({'context':context, "question": query})
        answer = answer.strip()
        return answer

    def run(self,query) ->Dict[str,str]:
        context,func_name,args = self.dispatch_function_call(query)
        logger.info(f"****function_call [{func_name}] result ****:\n{context}")
        answer = self._add_context_answer(query,context)
        ref_doc = f"本次回答基于使用工具[{func_name}]为您查询到结果:\n\n{context}\n\n"
        if answer: 
            return answer,ref_doc
        else:
            return REFUSE_ANSWER.format(func_name=func_name,args=args),''
        
    def run_with_func_args(self,query,func_name,args) ->Dict[str,str]:
        try:
            context = self._tool_call(func_name,**args)
            logger.info(f"****function_call [{func_name}] result ****:\n{context}")
            answer = self._add_context_answer(query,context)
            ref_doc = f"本次回答基于使用工具[{func_name}]为您查询到结果:\n\n{context}\n\n"
            if answer: 
                return answer,ref_doc
            else:
                return REFUSE_ANSWER.format(func_name=func_name,args=args),''
        except Exception as e:
            print(str(e))
            logger.info(str(e))
            return REFUSE_ANSWER.format(func_name=func_name,args=args),''
        
       
        


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
                f"Unknown exception, please check Lambda log for more details:{str(e)}"
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
    params = event.get('params')
    param_dict = params
    query = param_dict["query"]
    detection = param_dict["detection"]         
    use_bedrock = event.get('use_bedrock')
    
    region = os.environ.get('region')
    global BEDROCK_REGION
    BEDROCK_REGION = region
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
    



    agent_tools = AgentTools(api_schema=API_SCHEMA,llm=llm)
    agent_tools.register_tool(name='ec2_price',func=query_ec2_price)
    agent_tools.register_tool(name='service_org',func=service_org)

    func_name, func_params = None, None
    if detection:
        func_name = detection.get('func')
        func_params = detection.get('param')

    ## 已经从外面传入了识别出的意图和参数
    if func_name and func_params:
        if not agent_tools.check_tool(func_name):
            answer = f"对不起，该函数{func_name}还未实现"
            ref_doc = ''
        else:
            answer,ref_doc = agent_tools.run_with_func_args(query,func_name,func_params)
    else:
        answer,ref_doc = agent_tools.run(query)
    pattern = r'^根据[^，,]*[,|，]'
    answer = re.sub(pattern, "", answer)
    message = {"answer" : answer ,"ref_doc":ref_doc, "question": query }
    log_dict_str = json.dumps(message, ensure_ascii=False)
    logger.info(log_dict_str)
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body':message
    }