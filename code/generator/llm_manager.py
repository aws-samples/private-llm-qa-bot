import json
import boto3

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
    model_ids = [ item['modelId'] for item in response['modelSummaries'] if not item['modelId'].startswith('mistral') ]
    return model_ids

def get_all_private_llm(other_account_list=OTHER_ACCOUNT_LLM_ENDPOINTS):
    ret = {}

    # only get the llm endpoint from this account
    ssm = boto3.client('ssm')
    try:
        parameter = ssm.get_parameter(Name=SMM_KEY_AVAIL_LLM_ENDPOINTS, WithDecryption=False)
        ret=json.loads(parameter['Parameter']['Value'])
    except Exception as e:
        print("There is no llm endpoint existed.")

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
    try:
        ssm.put_parameter(
            Name=SMM_KEY_AVAIL_LLM_ENDPOINTS,
            Overwrite=True,
            Type='String',
            Value=ssm_val,
        )
    except Exception as e:
        return False
    
    return True 

def get_all_model_ids():
    private_llms = get_all_private_llm()
    bedrock_llms = get_all_bedrock_llm()
    model_ids = []
    model_ids += list(private_llms.keys())
    model_ids += bedrock_llms

    return model_ids