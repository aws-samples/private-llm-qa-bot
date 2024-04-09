import json
import os
import logging
import re
import argparse
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.prompts import PromptTemplate
# from langchain.llms import SagemakerEndpoint
from typing import Any, Dict, List, Union,Mapping, Optional, TypeVar, Union
# from langchain.chains import LLMChain
# from langchain.llms.bedrock import Bedrock
# from botocore.exceptions import ClientError
import boto3
import requests
from pydantic import BaseModel
from tools.get_price import query_ec2_price
from tools.service_org_demo import service_org
from generator.llm_wrapper import get_langchain_llm_model, invoke_model, format_to_message


logger = logging.getLogger()
logger.setLevel(logging.INFO)

credentials = boto3.Session().get_credentials()
lambda_client= boto3.client('lambda')
BEDROCK_REGION = None
BEDROCK_LLM_MODELID_LIST = {'claude-instant':'anthropic.claude-instant-v1',
                            'claude-v2':'anthropic.claude-v2:1',
                            'claude-v3-sonnet': 'anthropic.claude-3-sonnet-20240229-v1:0',
                            'claude-v3-haiku' : 'anthropic.claude-3-haiku-20240307-v1:0'}
REFUSE_ANSWER = '对不起, 根据{func_name}({args}),没有查询到您想要的信息，请您更具体的描述下您的要求.'

ERROR_ANSWER = """
            You are acting as a assistant.
            When a user asked a question:{query}
            a Large Language Model extracted the input arguments as {args}, and then use the args to call a function named:{func_name}.
            but it raised exception error in:
            <error>
            {error}
            </error>,
            please concisely response how to correct it, don't use code in response.
            Skip the preamble, go straight into the answer. Respond in the original language of user's question.
"""

FUNCTION_CALL_TEMPLATE = """here is a list of functions you can use, contains in <tools> tags

<tools>
{functions}
</tools>

Given you a task, you need to :
1. Decide which tool need to be chosen to solve the task, if no tool is chosen, you should reply “I don't know”
2. Once you decide a tool, please extract the required parameters and function name from the task. 
3. Please response in json format such as {{"name":"function name", "arguments":{{"x":1,"y":1}}}}, and enclose the response in <function_call></function_call> tag, 

Task: {task}"""

CONTEXT_TEMPLATE = """
        You are acting as an AWS assistant, please based on the context in <context></context>, answer user's question.
        Skip the preamble, go straight into the answer. if the context is empty,refuse to response politely.

        <context>
        {context}
        </context>

        Question: {question}
        """

Enhanced_TEMPLATE = """Human:You are acting as an AWS assistant, to response user's question in <query> tag, you call the corresponding API to get the response in <api_response>.

the user's query is:
<query>
{question}
</query>

<api_response>
{context}
</api_response>

Once again, the user's query is:

<query>
{question}
</query>

Please put your answer between <response> tags and follow below requirements:
1. Respond in the original language of the question.
2. If there is no relevant information in message, you should reply user that you can't find any information by his original question, don't say anything else.
3. if suggestion is provided and is not empty, you should suggest user to ask by referring suggestion. If no suggestion is empty, don't say anything else.
4. Do not begin with phrases like "API", skip the preamble, go straight into the answer. 
"""

class AgentTools(BaseModel):
    function_map: dict = {}
    llm: Any

    def register_tool(cls,name:str,func:Union[callable,str]) -> None:
        cls.function_map[name] = func

    def check_tool(cls,name:str) -> bool:
        return name in cls.function_map

    def _tool_call(cls,query,name,**args) -> Union[str,None]:
        func = cls.function_map.get(name)
        if callable(func):
            return func(**args) if func else None
        elif isinstance(func, str):
            # call lambda
            logger.info("call lambda:{}".format(func))
            payload = { "param" : args, "query": query }
            invoke_response = lambda_client.invoke(FunctionName=func,
                                                   InvocationType='RequestResponse',
                                                   Payload=json.dumps(payload))

            response_body = invoke_response['Payload']
            response_str = response_body.read().decode("unicode_escape")
            response_str = response_str.strip('"')

            return response_str

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

    def _add_error_answer(cls,query,func_name,args,error) ->str:
        prompt_template = PromptTemplate(
                template=ERROR_ANSWER,
                input_variables=["func_name",'args','error','query']
            )
        prompt = prompt_template.format(func_name=func_name, args=args, error=error, query=query)
        msg_list = [format_to_message(query=prompt)]
        ai_reply = invoke_model(llm=cls.llm, prompt=prompt, messages=msg_list)
        answer = ai_reply.content
        answer = answer.strip()
        return answer

    def _add_context_answer(cls,query,context) ->str:
        if not context:
            return None 
        prompt_template = PromptTemplate(
                template=Enhanced_TEMPLATE,
                input_variables=["context",'question']
            )
        prompt = prompt_template.format(context=context, question=query)
        # llmchain = LLMChain(llm=cls.llm,verbose=False,prompt = prompt)
        msg_list = [ format_to_message(query=prompt), {"role":"assistant", "content": "<response>"}]
        ai_reply = invoke_model(llm=cls.llm, prompt=prompt, messages=msg_list)
        answer = ai_reply.content
        logger.info(f'llm input:{prompt}')
        # answer = llmchain.run({'context':context, "question": query})
        answer = answer.replace('</response>','').strip()
        return answer
    
    def run_with_func_args(cls,query,func_name,args) ->Dict[str,str]:
        context= ''
        try:
            context = cls._tool_call(query, func_name, **args)
            logger.info(f"****function_call [{func_name}] result ****:\n{context}")
            answer = cls._add_context_answer(query,context)
            ref_doc = f"本次回答基于使用工具[{func_name}]为您查询到结果:\n\n{context}\n\n"
            if answer: 
                return answer,ref_doc
            else:
                return REFUSE_ANSWER.format(func_name=func_name,args=args),''
        except Exception as e:
            logger.info(str(e))
            answer = cls._add_error_answer(query,func_name,args,str(e))
            if answer: 
                ref_doc = f"本次回答基于使用工具[{func_name}]为您查询到结果:\n\n{context}\n\n"
                return answer,ref_doc
            else :
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
    intention = param_dict.get("intention")      
    detection = param_dict.get("detection")
    
    region = os.environ.get('region')
    agent_lambdas = os.environ.get('agent_tools', None)

    global BEDROCK_REGION
    BEDROCK_REGION = region
    llm_model_endpoint = os.environ.get('llm_model_endpoint','claude-v3-sonnet')
    logger.info("region:{}".format(region))
    logger.info("params:{}".format(params))
    logger.info("llm_model_endpoint:{}".format(llm_model_endpoint))
    
    parameters = {
        "temperature":0.01,
        "top_p":0.95,
        "stop": ["</response>"],
    }
    
    if llm_model_endpoint.startswith('claude') or llm_model_endpoint.startswith('anthropic'):
        model_id = BEDROCK_LLM_MODELID_LIST.get(llm_model_endpoint, BEDROCK_LLM_MODELID_LIST["claude-v3-sonnet"])
    else:
        model_id = llm_model_endpoint

    llm = get_langchain_llm_model(model_id, parameters, region, llm_stream=False)

    agent_tools = AgentTools(llm=llm)
    agent_tools.register_tool(name='ec2_price',func=query_ec2_price)
    agent_tools.register_tool(name='service_org',func=service_org)

    if agent_lambdas and len(agent_lambdas) > 0:
        agent_lambda_list = agent_lambdas.split(',')
        for lambda_name in agent_lambda_list:
            tool_name = lambda_name.replace('agent_tool_', '')
            logger.info("register lambda tool:{}".format(lambda_name))
            agent_tools.register_tool(name=tool_name,func=lambda_name)

    func_name, func_params = None, None

    logger.info("detection:{}".format(detection))
    if detection:
        func_name = detection.get('func')
        func_params = detection.get('param')
        logger.info("func_name:{}".format(func_name))
        logger.info("func_params:{}".format(func_params))


    ## 已经从外面传入了识别出的意图和参数
    if func_name and func_params:
        if not agent_tools.check_tool(func_name):
            answer = f"对不起，该函数{func_name}还未实现"
            ref_doc = ''
        else:
            answer,ref_doc = agent_tools.run_with_func_args(query,func_name,func_params)
    else:
        answer,ref_doc = f"对不起，未传入函数名称", ""

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

##for local test only
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default='查询g5.2x在美西2的价格')
    args = parser.parse_args()
    query = args.query
    event = {'params':{'query':query,'detection':{'func':'ec2_price','param':{'instance_type':'g5.2xlarge','region':'us-west-2'}}}}
    response = lambda_handler(event,{})
    print(response['body'])