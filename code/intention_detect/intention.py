import json
import os
import logging
from collections import Counter

from langchain.embeddings import SagemakerEndpointEmbeddings,BedrockEmbeddings
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain import PromptTemplate, SagemakerEndpoint
from typing import Any, Dict, List, Union,Mapping, Optional, TypeVar, Union
from langchain.chains import LLMChain
from langchain.llms.bedrock import Bedrock
from botocore.exceptions import ClientError
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth, helpers
import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

credentials = boto3.Session().get_credentials()
BEDROCK_EMBEDDING_MODELID_LIST = ["cohere.embed-multilingual-v3","cohere.embed-english-v3","amazon.titan-embed-text-v1"]

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

class ContentHandler(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, inputs: List[str], model_kwargs: Dict) -> bytes:
        instruction_zh = "为这个句子生成表示以用于检索相关文章："
        instruction_en = "Represent this sentence for searching relevant passages:"
        input_str = json.dumps({"inputs": inputs, "parameters":{}, "is_query":False, "instruction":instruction_en})
        return input_str.encode('utf-8')

    def transform_output(self, output: bytes) -> List[List[float]]:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["sentence_embeddings"]

class llmContentHandler(LLMContentHandler):
    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({'inputs': prompt,'history':[],**model_kwargs})
        return input_str.encode('utf-8')
    
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["outputs"]

def create_intention_prompt_templete():
    prompt_template = """{instruction}\n\n{fewshot}\n\nHuman: \"{query}\"，这个问题的提问意图是啥？可选项[{options}]\nAssistant: """

    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=['fewshot','query', 'instruction', 'options']
    )
    return PROMPT

    
@handle_error
def lambda_handler(event, context):
    
    embedding_endpoint = os.environ.get('embedding_endpoint')
    region = os.environ.get('region')
    aos_endpoint = os.environ.get('aos_endpoint')
    index_name = os.environ.get('index_name')
    query = event.get('query')
    fewshot_cnt = event.get('fewshot_cnt')
    use_bedrock = event.get('use_bedrock')
    llm_model_endpoint = os.environ.get('llm_model_endpoint')
    llm_model_name = event.get('llm_model_name', None)
    
    logger.info("embedding_endpoint: {}".format(embedding_endpoint))
    logger.info("region:{}".format(region))
    logger.info("aos_endpoint:{}".format(aos_endpoint))
    logger.info("index_name:{}".format(index_name))
    logger.info("fewshot_cnt:{}".format(fewshot_cnt))
    logger.info("llm_model_endpoint:{}".format(llm_model_endpoint))

    content_handler = ContentHandler()

    if embedding_endpoint in BEDROCK_EMBEDDING_MODELID_LIST :
        embeddings =  BedrockEmbeddings(region_name=region,model_id=embedding_endpoint) 
    else: 
       embeddings = SagemakerEndpointEmbeddings(
        endpoint_name=embedding_endpoint,
        region_name=region,
        content_handler=content_handler
    ) 

    auth = AWSV4SignerAuth(credentials, region)
        
    docsearch = OpenSearchVectorSearch(
        index_name=index_name,
        embedding_function=embeddings,
        opensearch_url="https://{}".format(aos_endpoint),
        http_auth = auth,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection
    )
    
    docs = docsearch.similarity_search_with_score(
        query=query, 
        k = fewshot_cnt,
        space_type="cosinesimil",
        vector_field="embedding",
        text_field="query",
        metadata_field='*'
    )

    docs_simple = [ {"query" : doc[0].page_content, "intention" : doc[0].metadata['intention'], "score":doc[1]} for doc in docs]

    intention_list = [doc['intention'] for doc in docs_simple ]
    intention_counter = Counter(intention_list)
    options = set(intention_list)
    options_str = ", ".join(options)

    instruction = "参考下列Example，回答下列选择题："
    examples = [ "Human: \"{}\"，这个问题的提问意图是啥？可选项[{}]\nAssistant: {}".format(doc['query'], options_str, doc['intention']) for doc in docs_simple ]
    fewshot_str = "{}\n{}\n{}".format("<example>", "\n\n".join(examples), "</example>")
    
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
            "top_p":1
        }
        
        model_id ="anthropic.claude-instant-v1" if llm_model_name == 'claude-instant' else "anthropic.claude-v2"
        llm = Bedrock(model_id=model_id, client=boto3_bedrock, model_kwargs=parameters)

    prompt_template = create_intention_prompt_templete()
    prompt = prompt_template.format(fewshot=fewshot_str, instruction=instruction, query=query, options=options_str)
    
    if len(options) == 1:
        logger.info("Notice: Only Single latent Intention detected.")
        answer = options.pop()
        log_dict = { "prompt" : prompt, "answer" : answer, "examples": docs_simple }
        log_dict_str = json.dumps(log_dict, ensure_ascii=False)
        logger.info(log_dict_str)
        return answer
        
    llmchain = LLMChain(llm=llm, verbose=False, prompt=prompt_template)
    answer = llmchain.run({'fewshot':fewshot_str, "instruction":instruction, "query":query, "options": options_str})
    answer = answer.strip()

    log_dict = { "prompt" : prompt, "answer" : answer , "examples": docs_simple }
    log_dict_str = json.dumps(log_dict, ensure_ascii=False)
    logger.info(log_dict_str)

    if answer not in options:
        answer = intention_counter.most_common(1)[0]
        for opt in options:
            if opt in answer:
                answer = opt
                break
        
    return answer
