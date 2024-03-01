import boto3
import os
import logging
from llm_wrapper import get_all_model_ids, get_langchain_llm_model
from enum import Enum
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

logger = logging.getLogger()
logger.setLevel(logging.INFO)

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

class LAMBDA_SUPPORT_METHOD(Enum):
    INVOKE_LLM = "INVOKE_LLM"
    GET_ALL_MODEL_IDS = "GET_ALL_MODEL_IDS"
    SAVE_PROMPT_TEMPLATE = "SAVE_PROMPT_TEMPLATE"

@handle_error
def lambda_handler(event, context):
    '''
        1. support get all model_ids
        2. call llm with model_id and prompt
    '''
    other_accounts = os.environ.get('other_accounts', '')
    region = os.environ.get('region')

    query = event.get('query', '')
    model_id = event.get('model_id', '')
    prompt = event.get('prompt', '')
    method = event.get('method', LAMBDA_SUPPORT_METHOD.GET_ALL_MODEL_IDS.value)

    logger.info("other_accounts: {}".format(other_accounts))
    logger.info("region:{}".format(region))
    logger.info("query:{}".format(query))
    logger.info("model_id:{}".format(model_id))
    logger.info("prompt:{}".format(prompt))


    if method == LAMBDA_SUPPORT_METHOD.GET_ALL_MODEL_IDS.value:
        all_model_ids = get_all_model_ids()
        return {"all_model_ids" : all_model_ids}
    elif method == LAMBDA_SUPPORT_METHOD.INVOKE_LLM.value:
        langchain_llm = get_langchain_llm_model(model_id)

        PROMPT = PromptTemplate(
            template=prompt, 
            input_variables=[]
        )

        llmchain = LLMChain(llm=langchain_llm, verbose=False, prompt=PROMPT)
        answer = llmchain.run({"role": "Jason"})
        return {"output": answer, "prompt" : prompt}
    elif method == LAMBDA_SUPPORT_METHOD.SAVE_PROMPT_TEMPLATE.value:
        raise RuntimeError(f"unsupported method - {method}.") 
    else:
        raise RuntimeError(f"unsupported method - {method}.") 