import boto3
import json
from typing import Any, Dict, List, Optional, Mapping
import logging
import copy
from langchain.pydantic_v1 import Extra, root_validator
from langchain.chains import LLMChain
from langchain.llms.base import LLM
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManagerForLLMRun
# from langchain.llms.bedrock import Bedrock
from .langchain_bedrock import Bedrock
from langchain.llms.sagemaker_endpoint import LLMContentHandler, SagemakerEndpoint
# from langchain.llms import SagemakerEndpoint
from langchain.prompts import PromptTemplate
from .llm_manager import get_all_private_llm, get_all_bedrock_llm

logger = logging.getLogger()
logger.setLevel(logging.INFO)

bedrock_llms = get_all_bedrock_llm()
private_llm = get_all_private_llm()

class StreamScanner:    
    def __init__(self):
        self.buff = io.BytesIO()
        self.read_pos = 0
        
    def write(self, content):
        self.buff.seek(0, io.SEEK_END)
        self.buff.write(content)
        
    def readlines(self):
        self.buff.seek(self.read_pos)
        for line in self.buff.readlines():
            if line[-1] != b'\n':
                self.read_pos += len(line)
                yield line[:-1]
                
    def reset(self):
        self.read_pos = 0

class SagemakerStreamContentHandler(LLMContentHandler):
    content_type: Optional[str] = "application/json"
    accepts: Optional[str] = "application/json"
    callbacks:BaseCallbackHandler
    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid
    
    def __init__(self,callbacks:BaseCallbackHandler,**kwargs) -> None:
        super().__init__(**kwargs)
        self.callbacks = callbacks
 
    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({'inputs': prompt,**model_kwargs})
        # logger.info(f'transform_input:{input_str}')
        return input_str.encode('utf-8')
    
    def transform_output(self, event_stream: Any) -> str:
        scanner = StreamScanner()
        text = ''
        for event in event_stream:
            scanner.write(event['PayloadPart']['Bytes'])
            for line in scanner.readlines():
                try:
                    resp = json.loads(line)
                    token = resp.get("outputs")['outputs']
                    text += token
                    for stop in STOP: ##如果碰到STOP截断
                        if text.endswith(stop):
                            self.callbacks.on_llm_end(None)
                            text = text.rstrip(stop)
                            return text
                    self.callbacks.on_llm_new_token(token)
                    # print(token, end='')
                except Exception as e:
                    # print(line)
                    continue
        self.callbacks.on_llm_end(None)
        return text

class llmContentHandler(LLMContentHandler):
    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({'inputs': prompt,'history':[], **model_kwargs})
        return input_str.encode('utf-8')
    
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["outputs"]

class SagemakerStreamEndpoint(LLM):
    endpoint_name: str = ""
    region_name: str = ""
    content_handler: LLMContentHandler
    model_kwargs: Optional[Dict] = None
    endpoint_kwargs: Optional[Dict] = None
    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid
    
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that AWS credentials to and python package exists in environment."""
        try:
            session = boto3.Session()
            values["client"] = session.client(
                "sagemaker-runtime", region_name=values["region_name"]
            )
        except Exception as e:
            raise ValueError(
                "Could not load credentials to authenticate with AWS client. "
                "Please check that credentials in the specified "
                "profile name are valid."
            ) from e
        return values
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"endpoint_name": self.endpoint_name},
            **{"model_kwargs": _model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "sagemaker_stream_endpoint"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        _model_kwargs = self.model_kwargs or {}
        _model_kwargs = {**_model_kwargs, **kwargs}
        _endpoint_kwargs = self.endpoint_kwargs or {}

        body = self.content_handler.transform_input(prompt, _model_kwargs)
        content_type = self.content_handler.content_type
        accepts = self.content_handler.accepts

        # send request
        try:
            response = self.client.invoke_endpoint_with_response_stream(
                EndpointName=self.endpoint_name,
                Body=body,
                ContentType=content_type,
                Accept=accepts,
                **_endpoint_kwargs,
            )
        except Exception as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")
        text = self.content_handler.transform_output(response["Body"])
        return text

def get_langchain_llm_from_sagemaker_endpoint(llm_model_endpoint, params, region, llm_stream, llm_callbacks):
    llm = None
    if llm_stream:
        # stream_callback should be the 1th of llm_callbacks
        llmcontent_handler = SagemakerStreamContentHandler(
            callbacks=llm_callbacks[0]
            )

        llm = SagemakerStreamEndpoint(
                endpoint_name=llm_model_endpoint, 
                region_name=region, 
                model_kwargs={'parameters': params},
                content_handler=llmcontent_handler,
                endpoint_kwargs={'CustomAttributes':'accept_eula=true'} ##for llama2
                )
    else:
        llmcontent_handler = llmContentHandler()
        llm = SagemakerEndpoint(
                endpoint_name=llm_model_endpoint, 
                region_name=region, 
                model_kwargs={'parameters': params},
                content_handler=llmcontent_handler,
                endpoint_kwargs={'CustomAttributes':'accept_eula=true'} ##for llama2
            )
    return llm

def get_langchain_llm_model(llm_model_id, params, region, llm_stream=False, llm_callbacks=[]):
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
                    if key == 'stop':
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
        llm = Bedrock(model_id=llm_model_id, 
            client=boto3_bedrock, 
            streaming=llm_stream, 
            callbacks=llm_callbacks,
            model_kwargs=adapt_parameters)                       

    elif llm_model_id in list(private_llm.keys()):
        llm_model_endpoint = private_llm[llm_model_id]
        llm = get_langchain_llm_from_sagemaker_endpoint(llm_model_endpoint, parameters, region, llm_stream, llm_callbacks)
    # it means that llm_model_id is sagemaker endpoint actually
    elif llm_model_id.endswith('endpoint'): 
        llm_model_endpoint = llm_model_id
        llm = get_langchain_llm_from_sagemaker_endpoint(llm_model_endpoint, parameters, region, llm_stream, llm_callbacks)
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

    INVOKE_MODEL_ID = 'anthropic.claude-3-sonnet-20240229-v1:0'
    llm2 = get_langchain_llm_model(INVOKE_MODEL_ID, params, REGION)
    llmchain = LLMChain(llm=llm2, verbose=False, prompt=prompt_templ)
    answer = llmchain.run({})
    print("claude3:" + answer)
    
    INVOKE_MODEL_ID = 'cohere.command-text-v14'
    llm = get_langchain_llm_model(INVOKE_MODEL_ID, params, REGION)
    llmchain = LLMChain(llm=llm, verbose=False, prompt=prompt_templ)
    answer = llmchain.run({})
    print("cohere:" + answer)

    INVOKE_MODEL_ID = 'anthropic.claude-v2'
    llm2 = get_langchain_llm_model(INVOKE_MODEL_ID, params, REGION)
    llmchain = LLMChain(llm=llm2, verbose=False, prompt=prompt_templ)
    answer = llmchain.run({})
    print("claude2:" + answer)

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

    INVOKE_MODEL_ID = 'qwen1.5_14B_GPTQ_INT4'
    prompt_templ = create_detect_prompt_templete()
    llm7 = get_langchain_llm_model(INVOKE_MODEL_ID, params, REGION)
    llmchain = LLMChain(llm=llm7, verbose=False, prompt=prompt_templ)
    answer = llmchain.run({})
    print("qwen1.5:" + answer)

    INVOKE_MODEL_ID = 'qwen15-14B-int4-2024-03-02-09-10-08-595-endpoint'
    prompt_templ = create_detect_prompt_templete()
    llm8 = get_langchain_llm_model(INVOKE_MODEL_ID, params, REGION)
    llmchain = LLMChain(llm=llm8, verbose=False, prompt=prompt_templ)
    answer = llmchain.run({})
    print("qwen1.5_endpoint:" + answer)
    
    # INVOKE_MODEL_ID = 'mistral.mistral-7b-instruct-v0:2'
    # llm4 = get_langchain_llm_model(INVOKE_MODEL_ID, params, REGION)
    # llmchain = LLMChain(llm=llm4, verbose=False, prompt=prompt_templ)
    # answer = llmchain.run({})
    # print("mistral:" + answer)