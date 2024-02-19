import json
import os
import logging
from collections import Counter

from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.prompts import PromptTemplate
from langchain.llms import SagemakerEndpoint
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


from typing import Any, Dict, List, Optional
from langchain.embeddings.base import Embeddings


class BedrockCohereEmbeddings(Embeddings):
    client: Any  #: :meta private:

    region_name: Optional[str] = None
    """The aws region e.g., `us-west-2`. Fallsback to AWS_DEFAULT_REGION env variable
    or region specified in ~/.aws/config in case it is not provided here.
    """

    credentials_profile_name: Optional[str] = None
    """The name of the profile in the ~/.aws/credentials or ~/.aws/config files, which
    has either access keys or role information specified.
    If not specified, the default credential profile or, if on an EC2 instance,
    credentials from IMDS will be used.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
    """

    model_id: str = "cohere.embed-multilingual-v3"
    """Id of the model to call, e.g., amazon.titan-e1t-medium, this is
    equivalent to the modelId property in the list-foundation-models api"""

    model_kwargs: Optional[Dict] = None
    """Key word arguments to pass to the model."""

    def __init__(self, region_name, model_id) -> None:
        self.region_name = region_name
        self.model_id = model_id
        self.client = boto3.client(service_name='bedrock-runtime', region_name=region_name)

    def embed_documents(
        self, texts: List[str], chunk_size: int = 1
    ) -> List[List[float]]:
        input_body = {}
        input_body["texts"] = texts
        input_body["input_type"] = 'search_document'
        #input_body["truncate"] = 'RIGHT'
        body = json.dumps(input_body)
        content_type = "application/json"
        accepts = "application/json"

        embeddings = []
        try:
            response = self.client.invoke_model(
                body=body,
                modelId=self.model_id,
                accept=accepts,
                contentType=content_type,
            )
            response_body = json.loads(response.get("body").read())
            embeddings = response_body.get("embeddings")
        except Exception as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        input_body = {}
        input_body["texts"] = [ text ]
        input_body["input_type"] = 'search_query'
        #input_body["truncate"] = 'RIGHT'
        body = json.dumps(input_body)
        content_type = "application/json"
        accepts = "application/json"

        embeddings = []
        try:
            response = self.client.invoke_model(
                body=body,
                modelId=self.model_id,
                accept=accepts,
                contentType=content_type,
            )
            response_body = json.loads(response.get("body").read())
            embeddings = response_body.get("embeddings")
        except Exception as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        return embeddings[0]

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

# def create_intention_prompt_templete():
#     prompt_template = """{instruction}\n\n{fewshot}\n\nHuman: \"{query}\"，这个问题的提问意图是啥？可选项[{options}]\nAssistant: """

#     PROMPT = PromptTemplate(
#         template=prompt_template, 
#         input_variables=['fewshot','query', 'instruction', 'options']
#     )
#     return PROMPT

def create_detect_prompt_templete():
    prompt_template = """Human:Here is a list of aimed functions:\n\n<api_schemas>{api_schemas}</api_schemas>\n\nYou should follow below examples to choose the corresponding function and params according to user's query\n\n<examples>{examples}</examples>\n\nAssistant:<query>{query}</query>\n<output>{prefix}"""

    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=['api_schemas','examples', 'query', 'prefix']
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
    llm_model_endpoint = os.environ.get('llm_model_endpoint', 'anthropic.claude-instant-v1')
    
    logger.info("embedding_endpoint: {}".format(embedding_endpoint))
    logger.info("region:{}".format(region))
    logger.info("aos_endpoint:{}".format(aos_endpoint))
    logger.info("index_name:{}".format(index_name))
    logger.info("fewshot_cnt:{}".format(fewshot_cnt))
    logger.info("llm_model_endpoint:{}".format(llm_model_endpoint))

    content_handler = ContentHandler()

    if embedding_endpoint in BEDROCK_EMBEDDING_MODELID_LIST :
        embeddings =  BedrockCohereEmbeddings(region_name=region, model_id=embedding_endpoint) 
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

    docs_simple = [ {"query" : doc[0].page_content, "detection" : doc[0].metadata['detection'], "api_schema" : doc[0].metadata['api_schema'], "score":doc[1]} for doc in docs]

    example_list = [ "<query>{}</query>\n<output>{}</output>".format(doc['query'], json.dumps(doc['detection'], ensure_ascii=False)) for doc in docs_simple ]

    api_schema_list = [ doc['api_schema'] for doc in docs_simple]

    options = set([ doc['detection'] for doc in docs_simple])

    if len(options) == 1:
        logger.info("Notice: Only Single latent Intention detected.")
        answer = options.pop()
        log_dict = { "answer" : answer, "examples": docs_simple }
        log_dict_str = json.dumps(log_dict, ensure_ascii=False)
        logger.info(log_dict_str)
        return answer

    api_schema_options = set(api_schema_list)
    api_schema_str = "<api_schema>\n{}\n</api_schema>".format(",\n".join(api_schema_options))
    example_list_str = "\n{}\n".format("\n".join(example_list))
    
    parameters = {
        "temperature": 0.01,
    }

    llm = None
    if llm_model_endpoint not in ["anthropic.claude-instant-v1", "anthropic.claude-v2"]:
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
            "max_tokens_to_sample": 50,
            "stop_sequences": ["</output>"],
            "temperature":0.01,
            "top_p":1
        }
        
        llm = Bedrock(model_id=llm_model_endpoint, client=boto3_bedrock, model_kwargs=parameters)

    prompt_template = create_detect_prompt_templete()
    prefix = """{"func":"""

    prompt = prompt_template.format(api_schemas=api_schema_str, examples=example_list_str, query=query, prefix=prefix)

    llmchain = LLMChain(llm=llm, verbose=False, prompt=prompt_template)
    answer = llmchain.run({"api_schemas":api_schema_str, "examples": example_list_str, "query":query, "prefix" : prefix})
    answer = prefix + answer.strip()

    log_dict = { "prompt" : prompt, "answer" : answer , "examples": docs_simple }
    log_dict_str = json.dumps(log_dict, ensure_ascii=False)
    logger.info(log_dict_str)

    # if answer not in options:
    #     answer = intention_counter.most_common(1)[0]
    #     for opt in options:
    #         if opt in answer:
    #             answer = opt
    #             break
    ret = {"func":"chat"}
    try:
        ret = json.loads(answer)
    except Exception as e:
        logger.info("Fail to detect function, caused by {}".format(str(e)))

    return ret
