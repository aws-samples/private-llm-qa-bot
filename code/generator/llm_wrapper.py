import boto3
import json
from typing import Any, Dict, List
import logging
import copy
from langchain.chains import LLMChain
from langchain.llms.bedrock import Bedrock
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.llms import SagemakerEndpoint
from langchain.prompts import PromptTemplate
from llm_manager import get_all_private_llm, get_all_bedrock_llm

logger = logging.getLogger()
logger.setLevel(logging.INFO)

bedrock_llms = get_all_bedrock_llm()
private_llm = get_all_private_llm()

class llmContentHandler(LLMContentHandler):
    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({'inputs': prompt,'history':[], **model_kwargs})
        return input_str.encode('utf-8')
    
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["outputs"]

def get_langchain_llm_model(llm_model_id, params, region):
    '''
        params keys should be in [temperature, max_tokens, top_p, top_k, stop]
    '''
    llm = None
    parameters = { item[0]:item[1] for item in params.items() if item[0] in ['temperature', 'max_tokens', 'top_p', 'top_k', 'stop']}

    if llm_model_id in bedrock_llms:
        boto3_bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name=region
        )

        # make sure its default value
        if bool(parameters) == False:
            parameters = {'temperature':0.1, 'max_tokens': 256, 'top_p': 0.8}

        adapt_parameters = copy.deepcopy(parameters)
        
        if llm_model_id.startswith('anthropic'):
            for key, value in parameters.items():
                if key in ['temperature', 'max_tokens', 'top_p', 'top_k', 'stop']:
                    if key == 'max_tokens':
                        adapt_parameters['max_tokens_to_sample'] = adapt_parameters.pop("max_tokens", None)
                    elif key == 'stop':
                        adapt_parameters['stop_sequences'] = adapt_parameters.pop("stop", None)
                    else:
                        adapt_parameters[key] = value
                else:
                    adapt_parameters.pop(key, None)
        elif llm_model_id.startswith('mistral'):
            for key, value in parameters.items():
                if key in ['temperature', 'max_tokens', 'top_p', 'top_k', 'stop']:
                    adapt_parameters[key] = value
                else:
                    adapt_parameters.pop(key, None)
        elif llm_model_id.startswith('meta'):
            for key, value in parameters.items():
                if key in ['max_tokens', 'temperature', 'top_p']:
                    if key == 'max_tokens':
                        adapt_parameters['max_gen_len'] = adapt_parameters.pop("max_tokens", None)
                    else:
                        adapt_parameters[key] = value
                else:
                    adapt_parameters.pop(key, None)
        elif llm_model_id.startswith('ai21'):
            for key, value in parameters.items():
                if key in ['max_tokens', 'temperature', 'top_p', 'stop']:
                    if key == 'max_tokens':
                        adapt_parameters['maxTokens'] = adapt_parameters.pop("max_tokens", None)
                    elif key == 'top_p':
                        adapt_parameters['topP'] = adapt_parameters.pop("top_p", None)
                    elif key == 'stop':
                        adapt_parameters['stopSequences'] = adapt_parameters.pop("stop", None)
                    else:
                        adapt_parameters[key] = value
                else:
                    adapt_parameters.pop(key, None)
        elif llm_model_id.startswith('cohere'):
            for key, value in parameters.items():
                if key in ['temperature', 'max_tokens', 'top_p', 'top_k', 'stop']:
                    if key == 'top_p':
                        adapt_parameters['p'] = adapt_parameters.pop("top_p", None)
                    elif key == 'top_k':
                        adapt_parameters['k'] = adapt_parameters.pop("top_k", None)
                    elif key == 'stop':
                        adapt_parameters['stop_sequences'] = adapt_parameters.pop("stop", None)
                    else:
                        adapt_parameters[key] = value
                else:
                    adapt_parameters.pop(key, None)
        elif llm_model_id.startswith('amazon'):
            for key, value in parameters.items():
                if key in ['temperature', 'max_tokens', 'top_p']:
                    if key == 'top_p':
                        adapt_parameters['topP'] = adapt_parameters.pop("top_p", None)
                    elif key == 'max_tokens':
                        adapt_parameters['maxTokenCount'] = adapt_parameters.pop("max_tokens", None)
                    elif key == 'stop':
                        adapt_parameters['stopSequences'] = adapt_parameters.pop("stop", None)
                    else:
                        adapt_parameters[key] = value
                else:
                    adapt_parameters.pop(key, None)
            
        logger.info("--------adapt_parameters------")
        logger.info(adapt_parameters)
        logger.info("--------adapt_parameters------")
        llm = Bedrock(model_id=llm_model_id, client=boto3_bedrock, streaming=False, model_kwargs=adapt_parameters) 

    elif llm_model_id in list(private_llm.keys()):
        llm_model_endpoint = private_llm[llm_model_id]
        llmcontent_handler = llmContentHandler()
        llm = SagemakerEndpoint(
                endpoint_name=llm_model_endpoint, 
                region_name=region, 
                model_kwargs={'parameters': parameters},
                content_handler=llmcontent_handler
            )
    else:
        raise RuntimeError(f"No Available Model - {llm_model_id}.") 

    return llm

if __name__ == "__main__":
    
    params = {}
    REGION='us-west-2'
    # print(get_all_private_llm())
    # llm_endpoint_regist('baichuan_13B_INT4', 'baichuan-13b-gptq2-2024-01-24-10-15-10-154-endpoint')
    # print(get_all_private_llm())
    # print(get_all_model_ids())

    def create_detect_prompt_templete():
        prompt_template = "你好啊，今天怎么样"
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=[]
        )
        return PROMPT

    prompt_templ = create_detect_prompt_templete()
    
    INVOKE_MODEL_ID = 'cohere.command-text-v14'
    llm = get_langchain_llm_model(INVOKE_MODEL_ID, params, REGION)
    llmchain = LLMChain(llm=llm, verbose=False, prompt=prompt_templ)
    answer = llmchain.run({})
    print("cohere:" + answer)

    INVOKE_MODEL_ID = 'anthropic.claude-v2'
    llm2 = get_langchain_llm_model(INVOKE_MODEL_ID, params, REGION)
    llmchain = LLMChain(llm=llm2, verbose=False, prompt=prompt_templ)
    answer = llmchain.run({})
    print("claude:" + answer)

    INVOKE_MODEL_ID = 'meta.llama2-13b-chat-v1'
    llm3 = get_langchain_llm_model(INVOKE_MODEL_ID, params, REGION)
    llmchain = LLMChain(llm=llm3, verbose=False, prompt=prompt_templ)
    answer = llmchain.run({})
    print("llama:" + answer)

    INVOKE_MODEL_ID = 'amazon.titan-text-express-v1'
    llm5 = get_langchain_llm_model(INVOKE_MODEL_ID, params, REGION)
    llmchain = LLMChain(llm=llm5, verbose=False, prompt=prompt_templ)
    answer = llmchain.run({})
    print("amazon:" + answer)

    INVOKE_MODEL_ID = 'ai21.j2-mid-v1'
    prompt_templ = create_detect_prompt_templete()
    llm6 = get_langchain_llm_model(INVOKE_MODEL_ID, params, REGION)
    llmchain = LLMChain(llm=llm6, verbose=False, prompt=prompt_templ)
    answer = llmchain.run({})
    print("ai21:" + answer)
    
    INVOKE_MODEL_ID = 'mistral.mistral-7b-instruct-v0:2'
    llm4 = get_langchain_llm_model(INVOKE_MODEL_ID, params, REGION)
    llmchain = LLMChain(llm=llm4, verbose=False, prompt=prompt_templ)
    answer = llmchain.run({})
    print("mistral:" + answer)