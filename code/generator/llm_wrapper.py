import boto3
import json
from typing import Any, Dict, List, Union,Mapping, Optional, TypeVar, Union
from langchain.chains import LLMChain
from langchain.llms.bedrock import Bedrock
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.llms import SagemakerEndpoint
from langchain.prompts import PromptTemplate

INVOKE_MODEL_ID = 'anthropic.claude-v2'
REGION = 'us-west-2'
SMM_KEY_AVAIL_LLM_ENDPOINTS = 'avail_llm_endpoints'
OTHER_ACCOUNT_LLM_ENDPOINTS = None

def get_all_bedrock_llm():
    bedrock = boto3.client(
        service_name='bedrock',
        region_name='us-west-2'
    )

    bedrock.list_foundation_models()

    response = bedrock.list_foundation_models(
        byOutputModality='TEXT',
        byInferenceType='ON_DEMAND'
    )
    model_ids = [ item['modelId'] for item in response['modelSummaries']]
    return model_ids

def get_all_private_llm(other_account_list=OTHER_ACCOUNT_LLM_ENDPOINTS):
    ret = {}

    # only get the llm endpoint from this account
    ssm = boto3.client('ssm')
    try:
        parameter = ssm.get_parameter(Name=SMM_KEY_AVAIL_LLM_ENDPOINTS, WithDecryption=False)
        ret=json.loads(parameter['Parameter']['Value'])
    except Exception as e:
        print(str(e))


    if type(other_account_list) == list:
        # get all of llm endpoint from other account
        pass
        
    return ret

def llm_endpoint_regist(model_id, model_endpoint):
    ssm = boto3.client('ssm')
    existed_llm_endpoints_dict=get_all_private_llm()

    append_llm_endpoint = {
        model_id: model_endpoint,
    }
    existed_llm_endpoints_dict.update(append_llm_endpoint)

    ssm_val = json.dumps(existed_llm_endpoints_dict)
    ssm.put_parameter(
        Name=SMM_KEY_AVAIL_LLM_ENDPOINTS,
        Overwrite=True,
        Type='String',
        Value=ssm_val,
    )

class llmContentHandler(LLMContentHandler):
    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({'inputs': prompt,'history':[], **model_kwargs})
        return input_str.encode('utf-8')
    
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["outputs"]

def get_all_model_ids():
    private_llms = get_all_private_llm()
    bedrock_llms = get_all_bedrock_llm()
    model_ids = []
    model_ids += list(private_llms.keys())
    model_ids += bedrock_llms

    return model_ids

def get_langchain_llm_model(llm_model_id):
    private_llm = get_all_private_llm()
    llm = None
    bedrock_llms = get_all_bedrock_llm()
    if llm_model_id in bedrock_llms:
        boto3_bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name=REGION
        )

        parameters = {
            "max_tokens_to_sample": 50,
            "stop_sequences": ["Human:"],
            "temperature":0.01,
            "top_p":1
        }
        
        llm = Bedrock(model_id=llm_model_id, client=boto3_bedrock, model_kwargs=parameters) 

    elif llm_model_id in list(private_llm.keys()):
        parameters = {
            "temperature": 0.01,
        }
        llm_model_endpoint = private_llm[llm_model_id]
        llmcontent_handler = llmContentHandler()
        llm = SagemakerEndpoint(
                endpoint_name=llm_model_endpoint, 
                region_name=REGION, 
                model_kwargs={'parameters':parameters},
                content_handler=llmcontent_handler
            )
    else:
        raise RuntimeError(f"No Available Model - {llm_model_id}.") 

    return llm

if __name__ == "__main__":
    print(get_all_private_llm())
    llm_endpoint_regist('baichuan_13B_INT4', 'baichuan-13b-gptq2-2024-01-24-10-15-10-154-endpoint')
    print(get_all_private_llm())

    print(get_all_model_ids())

    llm = get_langchain_llm_model(INVOKE_MODEL_ID)

    def create_detect_prompt_templete():
        prompt_template = "{role} 你好啊，今天怎么样"

        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=['role']
        )
        return PROMPT

    prompt_templ = create_detect_prompt_templete()
    
    llmchain = LLMChain(llm=llm, verbose=False, prompt=prompt_templ)
    answer = llmchain.run({"role": "Jason"})
    print(answer)