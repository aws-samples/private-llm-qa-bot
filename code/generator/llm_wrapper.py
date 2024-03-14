import boto3
import json
from typing import Any, Dict, List, Optional, Mapping
import logging
import copy
import io
from langchain.pydantic_v1 import Extra, root_validator
from langchain.chains import LLMChain
from langchain.llms.base import LLM
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.bedrock import Bedrock
from langchain_community.chat_models import BedrockChat
from langchain.llms.sagemaker_endpoint import LLMContentHandler, SagemakerEndpoint
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from .llm_manager import get_all_private_llm, get_all_bedrock_llm
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger()
logger.setLevel(logging.INFO)

bedrock_llms = get_all_bedrock_llm()
private_llm = get_all_private_llm()

STOP=["user:", "用户：", "用户:", '</response>']

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
            logger.info(f'for event in event_stream:')
            scanner.write(event['PayloadPart']['Bytes'])
            for line in scanner.readlines():
                logger.info(f'for {line} in scanner.readlines():')
                try:
                    resp = json.loads(line)
                    token = resp.get("outputs")['outputs']
                    text += token
                    logger.info(f"token: {token}")
                    logger.info(f"text: {text}")
                    self.callbacks.on_llm_new_token(token)
                    for stop in STOP: ##如果碰到STOP截断
                        if text.endswith(stop):
                            self.callbacks.on_llm_end(None)
                            text = text.rstrip(stop)
                            return text

                    logger.info(f'self.callbacks.on_llm_new_token({token})')
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

        llm = SagemakerEndpoint(
                endpoint_name=llm_model_endpoint, 
                region_name=region, 
                streaming=True,
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
            logger.info("--------adapt_parameters------")
            logger.info(adapt_parameters)
            logger.info("--------adapt_parameters------")
            llm = BedrockChat(model_id=llm_model_id, 
                client=boto3_bedrock, 
                streaming=llm_stream, 
                callbacks=llm_callbacks,
                model_kwargs=adapt_parameters) 
        else:
            if llm_model_id.startswith('mistral'):
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
            elif llm_model_id.startswith('mistral'):
                for key, value in parameters.items():
                    if key in ['temperature', 'max_tokens', 'top_p', 'top_k', 'stop']:
                        adapt_parameters[key] = value
                    else:
                        adapt_parameters.pop(key, None)
            else:
                raise RuntimeError(f"unsupported llm : {llm_model_id}")

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

def format_to_message(query:str, image_base64_list:List[str]=None, role:str = "user"):
    '''
    history format:
    "history": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "iVBORw..."
                    }
                },
                {
                    "type": "text",
                    "text": "What's in these images?"
                }
            ]
        }
    ]
    '''
    if image_base64_list:
        content = [{ "type": "image", "source": { "type": "base64", "media_type": "image/jpeg", "data": image_base64 }} for image_base64 in image_base64_list ]
        content.append({ "type": "text", "text": query })
        return { "role": role, "content": content }

    return {"role": role, "content": query }

def invoke_model(llm, prompt:str=None, messages:List[Dict]=[]) -> AIMessage:
    logger.info(f'invoke_model with input [prompt=>{prompt}; messages=>{json.dumps(messages)}]')
    ai_reply = None
    if isinstance(llm, BedrockChat):
        if messages:
            ai_reply = llm.invoke(input=messages)
        else:
            raise RuntimeError("No valid input for BedrockChat")
    elif isinstance(llm, Bedrock) or isinstance(llm, SagemakerEndpoint) or isinstance(llm, SagemakerStreamEndpoint):
        if prompt:
            answer = llm.invoke(input=prompt)
            logger.info(f'The result of invoke_model=> {answer}')
            ai_reply = AIMessage(answer)
        else:
            raise RuntimeError("No valid input for Bedrock/SagemakerEndpoint/SagemakerStreamEndpoint")
    else:
        raise RuntimeError("unsupported LLM type.")

    logger.info(f'[2]The result of invoke_model=> {ai_reply.content}')
    return ai_reply

if __name__ == "__main__":

    params = {'temperature':0.1, 'max_tokens':1024, 'top_p':0.8, 'top_k':10}
    REGION='us-west-2'
    # print(get_all_private_llm())
    # llm_endpoint_regist('baichuan_13B_INT4', 'baichuan-13b-gptq2-2024-01-24-10-15-10-154-endpoint')
    # print(get_all_private_llm())
    # print(get_all_model_ids())

    INVOKE_MODEL_ID = 'anthropic.claude-3-sonnet-20240229-v1:0'
    llm2 = get_langchain_llm_model(INVOKE_MODEL_ID, params, REGION)
    msg = format_to_message(query="你好啊，今天怎么样")
    ai_msg = invoke_model(llm2, messages=[msg])
    print("claude3:" + ai_msg.content)
    answer = llm2.invoke(input=[msg])
    print("claude3:" + answer.content)

    # INVOKE_MODEL_ID = 'anthropic.claude-v2'
    # llm2 = get_langchain_llm_model(INVOKE_MODEL_ID, params, REGION)
    # msg = format_to_message(query="你好啊，今天怎么样")
    # answer = llm2.invoke(input=[msg])
    # print("claude2:" + answer.content)

    INVOKE_MODEL_ID = 'cohere.command-text-v14'
    llm2 = get_langchain_llm_model(INVOKE_MODEL_ID, params, REGION)
    ai_msg = invoke_model(llm2, prompt="你好啊，今天怎么样")
    print("claude3:" + ai_msg.content)
    answer = llm2.invoke(input="你好啊，今天怎么样")
    print("cohere:" + str(answer))

    INVOKE_MODEL_ID = 'meta.llama2-13b-chat-v1'
    llm2 = get_langchain_llm_model(INVOKE_MODEL_ID, params, REGION)
    answer = llm2.invoke(input="你好啊，今天怎么样")
    print("llama:" + str(answer))

    INVOKE_MODEL_ID = 'amazon.titan-text-express-v1'
    llm2 = get_langchain_llm_model(INVOKE_MODEL_ID, params, REGION)
    answer = llm2.invoke(input="你好啊，今天怎么样")
    print("amazon:" + str(answer))

    INVOKE_MODEL_ID = 'ai21.j2-mid-v1'
    llm2 = get_langchain_llm_model(INVOKE_MODEL_ID, params, REGION)
    answer = llm2.invoke(input="你好啊，今天怎么样")
    print("ai21:" + str(answer))

    INVOKE_MODEL_ID = 'qwen1.5_14B_GPTQ_INT4'
    llm2 = get_langchain_llm_model(INVOKE_MODEL_ID, params, REGION)
    answer = llm2.invoke(input="你好啊，今天怎么样")
    print("qwen1.5:" + str(answer))

    INVOKE_MODEL_ID = 'qwen15-14B-int4-2024-03-02-09-10-08-595-endpoint'
    llm2 = get_langchain_llm_model(INVOKE_MODEL_ID, params, REGION)
    answer = llm2.invoke(input="你好啊，今天怎么样")
    print("qwen1.5_endpoint:" + str(answer))

    # INVOKE_MODEL_ID = 'mistral.mistral-7b-instruct-v0:2'
    # llm2 = get_langchain_llm_model(INVOKE_MODEL_ID, params, REGION)
    # answer = llm2.invoke(input="你好啊，今天怎么样")
    # print("mistral:" + str(answer))