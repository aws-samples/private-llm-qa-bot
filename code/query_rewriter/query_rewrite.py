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
import re
BEDROCK_LLM_MODELID_LIST = {'claude-instant':'anthropic.claude-instant-v1',
                            'claude-v2':'anthropic.claude-v2:1'}
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

def create_rewrite_prompt_templete():
    # prompt_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language. don't translate the chat history and input. \n\nChat History:\n{history}\nFollow Up Input: {cur_query}\nStandalone question:"""
    prompt_template = \
"""
Human: Given the following conversation in <conversation></conversation>.
<conversation>
{history}
</conversation>
please rephrase the latest user question to be a standalone question, respond in the original language of user's question, don't translate the chat history and user question.
if you don't understand the {role_a}'s question, or the question is not relevant to the conversation. please keep the orginal question.
Skip the preamble, don't explain, go straight into the answer. Please put the standalone question in <standalone_question> tag
Assistant: <standalone_question>
"""
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=['history','role_a']
    )
    return PROMPT

def create_check_implicit_info_prompt_template():
    prompt_template = """Human: Please determine is there any implicit information of the lastest user's uttrance in conversation. You should answer with 'Yes' or 'No' in <answer>. You should refer the provided examples below.

<examples>
<example>
<conversation>
user: Sagemaker相关问题应该联系谁？
bot: Bruce Lee
user: 那EMR的呢？
</conversation>
<answer>Yes</answer>
</example>
<example>
<conversation>
user: zero-etl在中国可用了吗？
bot: 还不可用
user: 中国区sagemaker有jumpstart吗
</conversation>
<answer>No</answer>
</example>
</examples>

Assistant: <conversation>\n{conversation}\n</conversation>\n<answer>"""
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=['conversation']
    )
    return PROMPT

def extract_content(content: str):
    pattern = r"<standalone_question>(.*?)</standalone_question>"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        extract_content = match.group(1)
        return extract_content.strip('"')
    else:
        return content.strip('"')
    
    
@handle_error
def lambda_handler(event, context):
    region = os.environ.get('region')
    llm_model_endpoint = os.environ.get('llm_model_endpoint')
    params = event.get('params')
    use_bedrock = True if llm_model_endpoint.startswith('anthropic') or llm_model_endpoint.startswith('claude') else False
    role_a = event.get('role_a', 'user')
    role_b = event.get('role_b', 'AI')
    
    logger.info("region:{}".format(region))
    logger.info("params:{}".format(params))
    logger.info("llm_model_endpoint:{}".format(llm_model_endpoint))
    logger.info("use_bedrock:{}".format(bool(use_bedrock)))

    param_dict = params
    query = param_dict["query"]
    history = param_dict["history"]

    history_with_role = [ "{}: {}".format(role_a if idx % 2 == 0 else role_b, item) for idx, item in enumerate(history) ]
    history_str = "\n".join(history_with_role)
    history_str += f'\n{role_a}: {query}'

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
            "max_tokens_to_sample": 100,
            "stop_sequences": ["\n\n", '</standalone_question>'],
            "temperature":0.01,
            "top_p":1
        }

        model_id = BEDROCK_LLM_MODELID_LIST.get(llm_model_endpoint, llm_model_endpoint)
        llm = Bedrock(model_id=model_id, client=boto3_bedrock, model_kwargs=parameters)

    prompt_template = create_rewrite_prompt_templete()
    prompt = prompt_template.format(history=history_str, role_a=role_a)
    
    llmchain = LLMChain(llm=llm, verbose=False, prompt=prompt_template)
    answer = llmchain.run({'history':history_str, "role_a":role_a})
    answer = answer.strip()

    log_dict = { "history" : history, "answer" : answer , "cur_query": query, "prompt":prompt, "model_id" : model_id }
    log_dict_str = json.dumps(log_dict, ensure_ascii=False)
    logger.info(log_dict_str)
        
    return extract_content(answer)
