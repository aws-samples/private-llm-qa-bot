import json
import logging
import time
import os
import re
from botocore import config
from botocore.exceptions import ClientError,EventStreamError
from datetime import datetime, timedelta
import pytz
import boto3
import time
import hashlib
import uuid
# from transformers import AutoTokenizer
from enum import Enum
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain.llms.sagemaker_endpoint import LLMContentHandler,SagemakerEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,ConversationalRetrievalChain,ConversationChain
from langchain.schema import BaseRetriever
from langchain.schema import Document
from langchain.llms.bedrock import Bedrock
from pydantic import BaseModel,Field
from langchain.pydantic_v1 import Extra, root_validator

import openai
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, Dict, List, Union,Mapping, Optional, TypeVar, Union
from langchain.schema import LLMResult
from langchain.llms.base import LLM
import io
import math
from enum import Enum
from boto3 import client as boto3_client

lambda_client= boto3.client('lambda')
dynamodb_client = boto3.resource('dynamodb')
credentials = boto3.Session().get_credentials()
region = boto3.Session().region_name
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, 'es', session_token=credentials.token)

DOC_INDEX_TABLE= 'chatbot_doc_index'
logger = logging.getLogger()
logger.setLevel(logging.INFO)
sm_client = boto3.client("sagemaker-runtime")
lambda_client = boto3_client('lambda')
# llm_endpoint = 'bloomz-7b1-mt-2023-04-19-09-41-24-189-endpoint'
chat_session_table = os.environ.get('chat_session_table')
QA_SEP = "=>"
A_Role="用户"
B_Role="AWSBot"
A_Role_en="user"
SYSTEM_ROLE_PROMPT = '你是云服务AWS的智能客服机器人AWSBot'
Fewshot_prefix_Q="问题"
Fewshot_prefix_A="回答"
RESET = '/rs'
openai_api_key = None
STOP=[f"\n{A_Role_en}", f"\n{A_Role}", f"\n{Fewshot_prefix_Q}", '</response>']
CHANNEL_RET_CNT = 10

BM25_QD_THRESHOLD_HARD_REFUSE = float(os.environ.get('bm25_qd_threshold_hard',15.0))
BM25_QD_THRESHOLD_SOFT_REFUSE = float(os.environ.get('bm25_qd_threshold_soft',20.0))
KNN_QQ_THRESHOLD_HARD_REFUSE = float(os.environ.get('knn_qq_threshold_hard',0.6))
KNN_QQ_THRESHOLD_SOFT_REFUSE = float(os.environ.get('knn_qq_threshold_soft',0.8))
KNN_QD_THRESHOLD_HARD_REFUSE = float(os.environ.get('knn_qd_threshold_hard',0.6))
KNN_QD_THRESHOLD_SOFT_REFUSE = float(os.environ.get('knn_qd_threshold_soft',0.8))

KNN_QUICK_PEFETCH_THRESHOLD = float(os.environ.get('knn_quick_prefetch_threshold',0.95))

INTENTION_LIST = os.environ.get('intention_list', "")

TOP_K = int(os.environ.get('TOP_K',4))
NEIGHBORS = int(os.environ.get('neighbors',0))
KNOWLEDGE_BASE_ID = os.environ.get('knowledge_base_id',None)

BEDROCK_EMBEDDING_MODELID_LIST = ["cohere.embed-multilingual-v3","cohere.embed-english-v3","amazon.titan-embed-text-v1"]
BEDROCK_LLM_MODELID_LIST = {'claude-instant':'anthropic.claude-instant-v1',
                            'claude-v2':'anthropic.claude-v2:1'}

boto3_bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name= os.environ.get('bedrock_region',region)
)

knowledgebase_client = boto3.client("bedrock-agent-runtime", region)

###记录跟踪日志，用于前端输出
class TraceLogger(BaseModel):
    logs:List[str] =  Field([])
    ref_docs:List[str] = Field([])
    wsclient:Any = Field()
    connectionId:str = Field()
    msgid:str=Field()
    stream:bool = Field()
    use_trace:bool=Field()
    hide_ref:bool=Field()
    class Config:
        extra = 'forbid'
    
    def postMessage(self,text:str) -> None:
        data = json.dumps({ 'msgid':self.msgid, 'role': "AI", 'text': {'content':f'{text}\n\n'},'connectionId':self.connectionId })
        try:
            self.wsclient.post_to_connection(Data = data.encode('utf-8'),  ConnectionId=self.connectionId)
        except Exception as e:
            logger.warning(str(e))
            
    def add_ref(self,text:str) -> None:
        self.ref_docs.append(text)

    def trace(self,text:str) -> None:
        if not self.use_trace:
            return
        self.logs.append(text)
        if self.stream:
            self.postMessage(text)
        
    ##ref doc排在llm输出answer之后，使用外部传入的stream参数控制是否需要推到ws
    def dump_refs(self,stream) -> List[str]:
        if stream and not self.use_trace and not self.hide_ref: ##当不使用trace，不隐藏ref时才推送
            for text in self.ref_docs:
                self.postMessage(text)
        return '\n\n'.join(self.ref_docs)
    
    def dump_logs_to_string(self) -> List[str]:
        return '\n\n'.join(self.logs)
    
TRACE_LOGGER = None

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

class CustomStreamingOutCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming. Only works with LLMs that support streaming."""
    def __init__(self,wsclient:str,msgid:str,connectionId:str ,model_name:str,hide_ref:bool,use_stream:bool, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.wsclient = wsclient
        self.connectionId = connectionId
        self.msgid = msgid
        self.model_name= model_name
        self.recall_knowledge = []
        self.hide_ref = hide_ref
        self.use_stream = use_stream

    def add_recall_knowledge(self,recall_knowledge):

        self.recall_knowledge = recall_knowledge

    def postMessage(self,data):
        try:
            self.wsclient.post_to_connection(Data = data.encode('utf-8'),  ConnectionId=self.connectionId)
        except Exception as e:
            pass
            # print (f'post {json.dumps(data)} to_wsconnection error:{str(e)}')

    def message_format(self,messages):
        """Format messages as ChatGPT who only accepts roles of ['system', 'assistant', 'user']"""
        return [
            {'role': 'assistant', 'content': msg['content']}
            if msg['role'] == 'AI'
            else {'role': 'user', 'content': msg['content']}
            for msg in messages
        ]
    
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        data = json.dumps({ 'msgid':self.msgid, 'role': "AI", 'text': {'content':token},'connectionId':self.connectionId})
        self.postMessage(data)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        if (not self.hide_ref) and self.use_stream:
            text = format_reference(self.recall_knowledge)
            data = json.dumps({ 'msgid':self.msgid, 'role': "AI", 'text': {'content':f'{text}'},'connectionId':self.connectionId })
            self.postMessage(data)

        

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        data = json.dumps({ 'msgid':self.msgid, 'role': "AI", 'text': {'content':str(error[0])+'[DONE]'},'connectionId':self.connectionId})
        self.postMessage(data)

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
       
class ContentHandler(EmbeddingsContentHandler):
    parameters = {
        "max_new_tokens": 50,
        "temperature": 0,
        "min_length": 10,
        "no_repeat_ngram_size": 2,
    }
    def transform_input(self, inputs: list[str], model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"inputs": inputs, **model_kwargs})
        return input_str.encode('utf-8')

    def transform_output(self, output: bytes) -> List[List[float]]:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["sentence_embeddings"]

class llmContentHandler(LLMContentHandler):
    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({'inputs': prompt,**model_kwargs})
        return input_str.encode('utf-8')
    
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["outputs"]

class CustomDocRetriever(BaseRetriever):
    embedding_model_endpoint :str
    aos_endpoint: str
    aos_index: str
        
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True
        
    @classmethod
    def from_endpoints(cls,embedding_model_endpoint:str, aos_endpoint:str, aos_index:str,):
        return cls(embedding_model_endpoint=embedding_model_endpoint,
                  aos_endpoint=aos_endpoint,
                  aos_index=aos_index)
    
    #this is for standard langchain interface
    def get_relevant_documents(self, query_input: str) -> List[Document]:
        recall_knowledge = None
        if KNOWLEDGE_BASE_ID:
            recall_knowledge = self.get_relevant_documents_from_bedrock(KNOWLEDGE_BASE_ID, query_input)
        else:
            recall_knowledge,_,_= self.get_relevant_documents_custom(query_input) 
        top_k_results = []
        for item in recall_knowledge:
            top_k_results.append(Document(page_content=item.get('doc')))
        return top_k_results
       
     ## kkn前置检索FAQ,，如果query非常相似，则返回作为cache
    def knn_quick_prefetch(self,query_input: str) -> List[Any]:
        global KNN_QUICK_PEFETCH_THRESHOLD
        start = time.time()
        query_embedding = get_vector_by_sm_endpoint(query_input, sm_client, self.embedding_model_endpoint)
        aos_client = OpenSearch(
                hosts=[{'host': self.aos_endpoint, 'port': 443}],
                http_auth = awsauth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection
            )
        opensearch_knn_respose = search_using_aos_knn(aos_client,query_embedding[0], self.aos_index,size=3)
        elpase_time = time.time() - start
        logger.info(f'runing time of quick_knn_fetch : {elpase_time:.3f}s')
        filter_knn_result = [item for item in opensearch_knn_respose if (item['score'] > KNN_QUICK_PEFETCH_THRESHOLD and item['doc_type'] == 'Question')]
        if len(filter_knn_result) :
            filter_knn_result.sort(key=lambda x:x['score'])
        return filter_knn_result

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError
    
    
    def add_neighbours_doc(self,client,opensearch_respose):
        docs = []
        docs_dict = {}
        for item in opensearch_respose:
            ## only apply to 'Paragraph','Sentence' type file
            if item['doc_type'] in ['Paragraph','Sentence'] and ( 
                item['doc_title'].endswith('.wiki.json') or 
                item['doc_title'].endswith('.blog.json') or 
                item['doc_title'].endswith('.txt') or 
                item['doc_title'].endswith('.docx') or
                item['doc_title'].endswith('.pdf')) :
                #check if has duplicate content in Paragraph/Sentence types
                key = f"{item['doc_title']}-{item['doc_category']}-{item['idx']}"
                if key not in docs_dict:
                    docs_dict[key] = item['idx']
                    doc = self.search_paragraph_neighbours(client,item['idx'],item['doc_title'],item['doc_category'],item['doc_type'])
                    docs.append({ **item, "doc": doc } )
            else:
                docs.append(item)
        return docs

    def search_paragraph_neighbours(self,client, idx, doc_title,doc_category,doc_type):
        query ={
            "query":{
                "bool": {
                "must": [
                    {
                    "terms": {
                        "idx": [i for i in range(idx-NEIGHBORS,idx+NEIGHBORS+1)]
                    }
                    },
                    {
                    "terms": {
                        "doc_title": [doc_title]
                    }
                    },
                    {
                  "terms": {
                    "doc_category": [doc_category]
                    }
                    },
                    {
                    "terms": {
                        "doc_type": [doc_type]
                    }
                    }
                ]
                }
            }
        }
        query_response = client.search(
            body=query,
            index=self.aos_index
        )
         ## the 'Sentence' type has mappings sentence:content:idx = n:1:1, so need to filter out the duplicate idx
        idx_dict = {}
        doc = ''
        for item in query_response["hits"]["hits"]:
            key = item['_source']['idx']
            if key not in idx_dict:
                idx_dict[key]=key
                doc += item['_source']['content']+'\n'            
        return doc

    ## 调用排序模型
    def rerank(self,query_input: str, docs: List[Any],sm_client,cross_model_endpoint):
        inputs = [query_input]*len(docs)
        response_model = sm_client.invoke_endpoint(
            EndpointName=cross_model_endpoint,
            Body=json.dumps(
                {
                    "inputs": inputs,
                    "docs": [item['doc'] for item in docs]
                }
            ),
            ContentType="application/json",
        )
        json_str = response_model['Body'].read().decode('utf8')
        json_obj = json.loads(json_str)
        scores = json_obj['scores']
        return scores
    
    def de_duplicate(self,docs):
        unique_ids = set()
        nodup = []
        for item in docs:
            doc_hash = hashlib.md5(str(item['doc']).encode('utf-8')).hexdigest()
            if doc_hash not in unique_ids:
                nodup.append(item)
                unique_ids.add(doc_hash)
        return nodup

    def get_relevant_documents_from_bedrock(self, knowledge_base_id:str, query_input:str):
        response = knowledgebase_client.retrieve(
            knowledgeBaseId=knowledge_base_id,
            retrievalQuery={
                'text': query_input
            },
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': TOP_K 
                }
            },
        )

        def remove_s3_prefix(s3_path):
            return '/'.join(s3_path.split('/', 3)[3:])

        ret = [ {'idx': 1, 'rank_score':0, 'doc_classify':'-', 'doc_author':'-', 'doc_category': '-', 'doc_type':'Paragraph', 'doc':item['content']['text'],'score':item['score'], 'doc_title': remove_s3_prefix(item['location']['s3Location']['uri']) } for item in response['retrievalResults']]

        return ret
    
    def get_relevant_documents_custom(self, query_input: str):
        global BM25_QD_THRESHOLD_HARD_REFUSE, BM25_QD_THRESHOLD_SOFT_REFUSE
        global KNN_QQ_THRESHOLD_HARD_REFUSE, KNN_QQ_THRESHOLD_SOFT_REFUSE
        global KNN_QD_THRESHOLD_HARD_REFUSE, KNN_QD_THRESHOLD_SOFT_REFUSE
        start = time.time()
        query_embedding = get_vector_by_sm_endpoint(query_input, sm_client, self.embedding_model_endpoint)
        aos_client = OpenSearch(
                hosts=[{'host': self.aos_endpoint, 'port': 443}],
                http_auth = awsauth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection
            )
        opensearch_knn_respose = search_using_aos_knn(aos_client,query_embedding[0], self.aos_index,size=CHANNEL_RET_CNT)
        elpase_time = time.time() - start
        logger.info(f'runing time of opensearch_knn : {elpase_time}s seconds')
        
        # 4. get AOS invertedIndex recall
        start = time.time()
        opensearch_query_response = aos_search(aos_client, self.aos_index, "doc", query_input,size=CHANNEL_RET_CNT)
        # logger.info(opensearch_query_response)
        elpase_time = time.time() - start
        logger.info(f'runing time of opensearch_query : {elpase_time}s seconds')

        # 5. combine these two opensearch_knn_respose and opensearch_query_response
        def combine_recalls(opensearch_knn_respose, opensearch_query_response):
            '''
            filter knn_result if the result don't appear in filter_inverted_result
            '''
            def get_topk_items(opensearch_knn_respose, opensearch_query_response, topk=1):

                opensearch_knn_nodup = []
                unique_ids = set()
                for item in opensearch_knn_respose:
                    doc_hash = hashlib.md5(str(item['doc']).encode('utf-8')).hexdigest()
                    if doc_hash not in unique_ids:
                        opensearch_knn_nodup.append(item)
                        unique_ids.add(doc_hash)
                
                opensearch_bm25_nodup = []
                for item in opensearch_query_response:
                    doc_hash = hashlib.md5(str(item['doc']).encode('utf-8')).hexdigest()
                    if doc_hash not in unique_ids:
                        opensearch_bm25_nodup.append(item)
                        doc_hash = hashlib.md5(str(item['doc']).encode('utf-8')).hexdigest()
                        unique_ids.add(doc_hash)

                opensearch_knn_nodup.sort(key=lambda x: x['score'])
                opensearch_bm25_nodup.sort(key=lambda x: x['score'])
                
                half_topk = math.ceil(topk/2) 
    
                kg_combine_result = [ item for item in opensearch_knn_nodup[-1*half_topk:]]
                knn_kept_doc = [ item['id'] for item in opensearch_knn_nodup[-1*half_topk:] ]

                bm25_count = 0
                for item in opensearch_bm25_nodup[::-1]:
                    if item['id'] not in knn_kept_doc:
                        kg_combine_result.append(item)
                        bm25_count += 1
                    if bm25_count+len(knn_kept_doc) >= topk:
                        break
                ##继续填补不足的召回
                step_knn = 0
                step_bm25 = 0
                while topk - len(kg_combine_result)>0:
                    if len(opensearch_knn_nodup) > half_topk and len(opensearch_knn_nodup[-1*half_topk-1-step_knn:-1*half_topk-step_knn]) > 0:
                        kg_combine_result += [item for item in opensearch_knn_nodup[-1*half_topk-1-step_knn:-1*half_topk-step_knn]]
                        kg_combine_result.sort(key=lambda x: x['score'])
                        step_knn += 1
                    elif len(opensearch_bm25_nodup) > half_topk and len(opensearch_bm25_nodup[-1*half_topk-1-step_bm25:-1*half_topk-step_bm25]) >0:
                        kg_combine_result += [item for item in opensearch_bm25_nodup[-1*half_topk-1-step_bm25:-1*half_topk-step_bm25]]
                        step_bm25 += 1
                    else:
                        break

                return kg_combine_result
            
            ret_content = get_topk_items(opensearch_knn_respose, opensearch_query_response, TOP_K)
            logger.info(f'get_topk_items:{len(ret_content)}')
            return ret_content
        
        filter_knn_result = [ item for item in opensearch_knn_respose if (item['score'] > KNN_QQ_THRESHOLD_HARD_REFUSE and item['doc_type'] == 'Question') or 
                              (item['score'] > KNN_QD_THRESHOLD_HARD_REFUSE and item['doc_type'] == 'Paragraph') or
                              (item['score'] > KNN_QQ_THRESHOLD_HARD_REFUSE and item['doc_type'] == 'Sentence')]
        filter_inverted_result = [ item for item in opensearch_query_response if item['score'] > BM25_QD_THRESHOLD_HARD_REFUSE ]


        ##是否使用rerank
        cross_model_endpoint = os.environ.get('cross_model_endpoint',None)
        if cross_model_endpoint:
            all_docs = filter_knn_result+filter_inverted_result

            ###to do 去重
            all_docs = self.de_duplicate(all_docs)
            if all_docs:
                scores = self.rerank(query_input, all_docs,sm_client,cross_model_endpoint)
                ##sort by scores
                sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=False)
                recall_knowledge = [{**all_docs[idx],'rank_score':scores[idx] } for idx in sorted_indices[-TOP_K:]]
            else:
                recall_knowledge = []

        else:
            recall_knowledge = combine_recalls(filter_knn_result, filter_inverted_result)
            recall_knowledge = [{**doc,'rank_score':0 } for doc in recall_knowledge]

        ##如果是段落类型，添加临近doc
        recall_knowledge = self.add_neighbours_doc(aos_client,recall_knowledge)

        return recall_knowledge,opensearch_knn_respose,opensearch_query_response


class ReplyStratgy(Enum):
    LLM_ONLY = 1
    WITH_LLM = 2
    HINT_LLM_REFUSE = 3
    RETURN_OPTIONS = 4
    SAY_DONT_KNOW = 5
    AGENT = 6
    OTHER = 7


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

def detect_intention(query, fewshot_cnt=5):
    msg = {"fewshot_cnt":fewshot_cnt, "query": query}
    invoke_response = lambda_client.invoke(FunctionName="Detect_Intention",
                                           InvocationType='RequestResponse',
                                           Payload=json.dumps(msg))
    response_body = invoke_response['Payload']

    response_str = response_body.read().decode("unicode_escape")
    response_str = response_str.strip('"')

    return json.loads(response_str)

def rewrite_query(query, session_history, round_cnt=2):
    logger.info(f"session_history {str(session_history)}")
    if len(session_history) == 0:
        return query

    history = []
    for item in session_history[-1 * round_cnt:]:
        history.append(item[0])
        history.append(item[1])

    msg = {
      "params": {
        "history": history,
        "query": query
      }
    }
    response = lambda_client.invoke(FunctionName="Query_Rewrite",
                                           InvocationType='RequestResponse',
                                           Payload=json.dumps(msg))
    response_body = response['Payload']
    response_str = response_body.read().decode("unicode_escape")

    return response_str.strip('"')

def chat_agent(query, detection, use_bedrock="True"):

    msg = {
      "params": {
        "query": query,
        "detection": detection 
      },
      "use_bedrock" : use_bedrock,
      "llm_model_name" : "claude-v2"
    }
    response = lambda_client.invoke(FunctionName="Chat_Agent",
                                           InvocationType='RequestResponse',
                                           Payload=json.dumps(msg))
    payload_json = json.loads(response.get('Payload').read())
    body = payload_json['body']
    answer = body['answer']
    ref_doc = body['ref_doc']

    return answer,ref_doc

def is_chinese(string):
    for char in string:
        if '\u4e00' <= char <= '\u9fff':
            return True

    return False


def get_embedding_bedrock(texts,model_id):
    provider = model_id.split(".")[0]
    if provider == "cohere":
        body = json.dumps({
            "texts": [texts] if isinstance(texts, str) else texts,
            "input_type": "search_document"
        })
    else:
        # includes common provider == "amazon"
        body = json.dumps({
            "inputText": texts if isinstance(texts, str) else texts[0],
        })
    bedrock_resp = boto3_bedrock.invoke_model(
            body=body,
            modelId=model_id,
            accept="application/json",
            contentType="application/json"
        )
    response_body = json.loads(bedrock_resp.get('body').read())
    if provider == "cohere":
        embeddings = response_body['embeddings']
    else:
        embeddings = [response_body['embedding']]
    return embeddings


# AOS
def get_vector_by_sm_endpoint(questions, sm_client, endpoint_name):
    if endpoint_name in BEDROCK_EMBEDDING_MODELID_LIST:
        return get_embedding_bedrock(questions,endpoint_name)

    parameters = {
    }

    instruction_zh = "为这个句子生成表示以用于检索相关文章："
    instruction_en = "Represent this sentence for searching relevant passages:"

    if isinstance(questions, str):
        instruction = instruction_zh if is_chinese(questions) else instruction_en
    else:
        instruction = instruction_zh

    response_model = sm_client.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=json.dumps(
            {
                "inputs": questions,
                "parameters": parameters,
                "is_query" : False,
                "instruction" :  instruction
            }
        ),
        ContentType="application/json",
    )
    json_str = response_model['Body'].read().decode('utf8')
    json_obj = json.loads(json_str)
    embeddings = json_obj['sentence_embeddings']
    return embeddings

def search_using_aos_knn(client, q_embedding, index, size=10):

    #Note: 查询时无需指定排序方式，最临近的向量分数越高，做过归一化(0.0~1.0)
    #精准Knn的查询语法参考 https://opensearch.org/docs/latest/search-plugins/knn/knn-score-script/
    #模糊Knn的查询语法参考 https://opensearch.org/docs/latest/search-plugins/knn/approximate-knn/
    #这里采用的是模糊查询
    query = {
        "size": size,
        "query": {
            "knn": {
                "embedding": {
                    "vector": q_embedding,
                    "k": size
                }
            }
        }
    }
    opensearch_knn_respose = []
    query_response = client.search(
        body=query,
        index=index
    )
    opensearch_knn_respose = [{'idx':item['_source'].get('idx',1),'doc_category':item['_source']['doc_category'],'doc_classify':item['_source']['doc_classify'],'doc_title':item['_source']['doc_title'],'id':item['_id'],'doc':item['_source']['content'],"doc_type":item["_source"]["doc_type"],"score":item["_score"],'doc_author': item['_source']['doc_author'], 'doc_meta': item['_source']['doc_meta']}  for item in query_response["hits"]["hits"]]
    return opensearch_knn_respose
    


def aos_search(client, index_name, field, query_term, exactly_match=False, size=10):
    """
    search opensearch with query.
    :param host: AOS endpoint
    :param index_name: Target Index Name
    :param field: search field
    :param query_term: query term
    :return: aos response json
    """
    if not isinstance(client, OpenSearch):   
        client = OpenSearch(
            hosts=[{'host': client, 'port': 443}],
            http_auth = awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )
    query = None
    if exactly_match:
        query =  {
            "query" : {
                "match_phrase":{
                    "doc": {
                        "query": query_term,
                        "analyzer": "ik_smart"
                      }
                }
            }
        }
    else:
        query = {
            "size": size,
            "query": {
                "bool": {
                    "should": [{
                            "bool": {
                                "must": [{
                                        "term": {
                                            "doc_type": "Question"
                                        }
                                    },
                                    {
                                        "match": {
                                            "content": query_term
                                        }
                                    }
                                ]
                            }
                        },
                        {
                            "bool": {
                                "must": [{
                                        "term": {
                                            "doc_type": "Paragraph"
                                        }
                                    },
                                    {
                                        "match": {
                                            "content": query_term
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            },
            "sort": [{
                "_score": {
                    "order": "desc"
                }
            }]
        }
    query_response = client.search(
        body=query,
        index=index_name
    )

    if exactly_match:
        result_arr = [ {'idx':item['_source'].get('idx',0),'doc_category':item['_source']['doc_category'],'doc_classify':item['_source']['doc_classify'],'doc_title':item['_source']['doc_title'],'id':item['_id'],'doc': item['_source']['content'], 'doc_type': item['_source']['doc_type'], 'score': item['_score'],'doc_author': item['_source']['doc_author'], 'doc_meta': item['_source']['doc_meta']} for item in query_response["hits"]["hits"]]
    else:
        result_arr = [ {'idx':item['_source'].get('idx',0),'doc_category':item['_source']['doc_category'],'doc_classify':item['_source']['doc_classify'],'doc_title':item['_source']['doc_title'],'id':item['_id'],'doc':item['_source']['content'], 'doc_type': item['_source']['doc_type'], 'score': item['_score'],'doc_author': item['_source']['doc_author'], 'doc_meta': item['_source']['doc_meta']} for item in query_response["hits"]["hits"]]
    return result_arr

def delete_session(session_id,user_id):
    # dynamodb = boto3.resource('dynamodb')
    table = dynamodb_client.Table(chat_session_table)
    try:
        table.delete_item(
        Key={
            'session-id': session_id,
            'user_id':user_id
        })
    except Exception as e:
        logger.info(f"delete session failed {str(e)}")

        
def get_session(session_id):

    table_name = chat_session_table
    # dynamodb = boto3.resource('dynamodb')

    # table name
    table = dynamodb_client.Table(table_name)
    operation_result = ""
    try:
        response = table.get_item(Key={'session-id': session_id})
        if "Item" in response.keys():
        # print("****** " + response["Item"]["content"])
            operation_result = json.loads(response["Item"]["content"])
        else:
            # print("****** No result")
            operation_result = ""
        return operation_result
    except Exception as e:
        logger.info(f"get session failed {str(e)}")
        return ""




# param:    session_id
#           question
#           answer
# return:   success
#           failed
def update_session(session_id,msgid, question, answer, intention):

    table_name = chat_session_table
    # dynamodb = boto3.resource('dynamodb')

    # table name
    table = dynamodb_client.Table(table_name)
    operation_result = ""

    response = table.get_item(Key={'session-id': session_id})

    if "Item" in response.keys():
        # print("****** " + response["Item"]["content"])
        chat_history = json.loads(response["Item"]["content"])
    else:
        # print("****** No result")
        chat_history = []

    chat_history.append([question, answer, intention,msgid])
    content = json.dumps(chat_history,ensure_ascii=False)

    # inserting values into table
    response = table.put_item(
        Item={
            'session-id': session_id,
            'content': content
        }
    )

    if "ResponseMetadata" in response.keys():
        if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            operation_result = "success"
        else:
            operation_result = "failed"
    else:
        operation_result = "failed"

    return operation_result

def enforce_stop_tokens(text: str, stop: List[str]) -> str:
    """Cut off the text as soon as any stop words occur."""
    if stop is None:
        return text
    
    return re.split("|".join(stop), text)[0]

def format_knowledges(recalls):
    knowledges = []
    multi_choice_field = []
    meta_dict = {}
    for idx, item in enumerate(recalls):
        if len(item['doc_meta']) > 0:
            meta_obj = json.loads(item['doc_meta'])
            for k, v in meta_obj.items():
                if k in meta_dict.keys() and meta_dict[k] != v:
                    multi_choice_field.append(k)
                else:
                    meta_dict[k] = v
            item_obj = { "meta" : meta_obj, 'text': item['doc']}
            content = json.dumps(item_obj, ensure_ascii=False)
            item_str = f"""<item index="{idx+1}">{content}</item>"""
        else:
            item_obj = {'text': item['doc']}
            content = json.dumps(item_obj, ensure_ascii=False)
            item_str = f"""<item index="{idx+1}">{content}</item>"""
        knowledges.append(item_str)

    context_str = "\n".join(knowledges)
    return context_str, set(multi_choice_field)


def get_question_history(inputs) -> str:
    res = []
    for human, _ in inputs:
        res.append(f"{human}\n")
    return "\n".join(res)

def get_qa_history(inputs) -> str:
    res = []
    for human, ai in inputs:
        res.append(f"{human}:{ai}\n")
    return "\n".join(res)

def get_chat_history(inputs) -> str:
    res = []
    for human, ai in inputs:
        res.append(f"{A_Role}:{human}\n{B_Role}:{ai}")
    return "\n".join(res)

def create_baichuan_prompt_template(prompt_template):
    #template_1 = '以下context内的文本内容为背景知识：\n<context>\n{context}\n</context>\n请根据背景知识, 回答这个问题：{question}'
    #template_2 = '这是原始问题: {question}\n已有的回答: {existing_answer}\n\n现在context内的还有一些文本内容，（如果有需要）你可以根据它们完善现有的回答。\n<context>\n{context}\n</context>\n请根据新的文段，进一步完善你的回答。'
    if prompt_template == '':
        prompt_template_zh = """{system_role_prompt} {role_bot}\n以下context内的文本内容为背景知识:\n<context>\n{chat_history}{context}\n</context>\n请根据背景知识, 回答这个问题,如果context内的文本内容为空，则回答不知道.\n{question}"""
    else:
        prompt_template_zh = prompt_template
    PROMPT = PromptTemplate(
        template=prompt_template_zh,
        partial_variables={'system_role_prompt':SYSTEM_ROLE_PROMPT},
        input_variables=["context",'question','chat_history','role_bot']
    )
    return PROMPT

def create_soft_refuse_template(prompt_template):
    if prompt_template == '':
        # prompt_template_zh = """{system_role_prompt} {role_bot}\n请根据反引号中的内容提取相关信息回答问题:\n```\n{chat_history}{context}\n```\n如果反引号中信息不相关,则回答不知道.\n用户:{question}"""
        prompt_template_zh = \
"""{system_role_prompt}{role_bot}请根据以下的知识，回答用户的问题。
<context>
{context}
</context> 
如果知识中的内容的包含markdown格式的内容，如参考图片，示意图，链接等，请尽可能利用并按markdown格式输出参考图片，示意图，链接。请严格基于跟问题相关的知识来回答问题，不要随意发挥和编造答案。请简洁有条理的回答，如果知识内容为空或者跟问题不相关，则回答不知道。
前几轮的聊天记录如下，如果有需要请参考以下的记录。
<chat_history>
{chat_history} 
</chat_history>
Skip the preamble, go straight into the answer.
用户问:{question} """
    else:
        prompt_template_zh = prompt_template
    PROMPT = PromptTemplate(
        template=prompt_template_zh,
        partial_variables={'system_role_prompt':SYSTEM_ROLE_PROMPT},
        input_variables=["context",'question','chat_history','role_bot']
    )
    return PROMPT

def create_qa_prompt_templete(prompt_template):
    if prompt_template == '':
        prompt_template_zh = \
"""Human: {system_role_prompt}{role_bot}, here is a query:
{chat_history}
<query>
{question}
</query>

Below may contains some relevant information to the query:

<information>
{context}
</information>

Once again, the user's query is:

<query>
{question}
</query>

Please put your answer between <response> tags and follow below requirements:
- Respond in the original language of the question.
- Maintain a friendly and conversational tone. 
- Skip the preamble, go straight into the answer. Don't say anything else.
{ask_user_prompt}
Assistant: <response>"""
    else:
        prompt_template_zh = prompt_template
    PROMPT = PromptTemplate(
        template=prompt_template_zh,
        partial_variables={'system_role_prompt':SYSTEM_ROLE_PROMPT},
        input_variables=["context",'question','chat_history', 'role_bot', 'ask_user_prompt']
    )
    return PROMPT

# def create_qa_prompt_templete(prompt_template):
#     if prompt_template == '':
#         #prompt_template_zh = """{system_role_prompt} {role_bot}\n请根据反引号中的内容提取相关信息回答问题:\n```\n{chat_history}{context}\n```\n如果反引号中的内容为空,则回答不知道.\n用户:{question}"""
#         prompt_template_zh = \
# """{system_role_prompt}{role_bot}请根据以下的知识，回答用户的问题。
# <context>
# {context}
# </context> 
# 如果知识中的内容的包含markdown格式的内容，如参考图片，示意图，链接等，请尽可能利用并按markdown格式输出参考图片，示意图，链接。请严格基于跟问题相关的知识来回答问题，不要随意发挥和编造答案。请简洁有条理的回答，如果知识内容为空或者跟问题不相关，则回答不知道。
# 前几轮的聊天记录如下，如果有需要请参考以下的记录。
# <chat_history>
# {chat_history} 
# </chat_history>
# Skip the preamble, go straight into the answer.
# 用户问:{question} 
# """
#     else:
#         prompt_template_zh = prompt_template
#     PROMPT = PromptTemplate(
#         template=prompt_template_zh,
#         partial_variables={'system_role_prompt':SYSTEM_ROLE_PROMPT},
#         input_variables=["context",'question','chat_history','role_bot']
#     )
#     return PROMPT

def create_chat_prompt_templete(prompt_template='', llm_model_name='claude'):
    PROMPT = None
    if llm_model_name.startswith('claude'):
        prompt_template_zh = """Human: {system_role_prompt}{role_bot}. Your goal is to be kind and helpful to users.
You should maintain a friendly customer service tone.
Here are some important rules for the interaction:
- Always stay in character, as {role_bot}
- If you are unsure how to respond, say “Sorry, I didn’t understand that. Could you repeat the question?”
- Be polite and patient
Here is the conversation history (between the user and you) prior to the question. It could be empty if there is no history:
<history> {chat_history} </history>
Here is the user’s question: <question> {question} </question>
How do you respond to the user’s question?
Think about your answer first before you respond. Put your response in <response></response> tags.
Assistant: <response>"""
        PROMPT = PromptTemplate(
            template=prompt_template_zh, 
            partial_variables={'system_role_prompt':SYSTEM_ROLE_PROMPT},
            input_variables=['question', 'chat_history','role_bot']
        )
    else:
        if prompt_template == '':
            prompt_template_zh = """Human:{system_role_prompt}{role_bot}\n{chat_history}\n\n{question}"""
        else:
            prompt_template_zh = prompt_template.replace('{context}','') ##remove{context}
        PROMPT = PromptTemplate(
            template=prompt_template_zh, 
            partial_variables={'system_role_prompt':SYSTEM_ROLE_PROMPT},
            input_variables=['question','chat_history','role_bot']
        )
    return PROMPT

def create_assist_prompt_templete(prompt_template='', llm_model_name='claude'):
    PROMPT = None
    if llm_model_name.startswith('claude'):
        prompt_template_zh = """Human: You will be acting as an AI Assistant named {role_bot}. Your goal is to answer users' question and help them finish their work.
You should maintain a friendly customer service tone.
Here are some important rules for the interaction:
- Always stay in character, as {role_bot}
- If you are unsure how to respond, say “Sorry, I didn’t understand that. Could you repeat the question?”

Here is the conversation history (between the user and you) prior to the question. It could be empty if there is no history:
<history> {chat_history} </history>
Here is the user’s question: <question> {question} </question>
How do you respond to the user’s question?
Think about your answer first before you respond. Put your response in <response></response> tags.
Assistant: <response>"""
        PROMPT = PromptTemplate(
            template=prompt_template_zh, 
            input_variables=['question', 'chat_history','role_bot']
        )
    else:
        if prompt_template == '':
            prompt_template_zh = """Human:{question}\n"""
        else:
            prompt_template_zh = prompt_template.replace('{context}','') ##remove{context}
        PROMPT = PromptTemplate(
            template=prompt_template_zh, 
            input_variables=['question','chat_history','role_bot']
        )
    return PROMPT

def get_bedrock_aksk(secret_name='chatbot_bedrock', region_name = os.environ.get('bedrock_region',"us-west-2") ):
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    # Decrypts secret using the associated KMS key.
    secret = json.loads(get_secret_value_response['SecretString'])
    return secret['BEDROCK_ACCESS_KEY'],secret['BEDROCK_SECRET_KEY']


def format_reference(recall_knowledge):
    if not recall_knowledge:
        return ''
    text = '\n```json\n#Reference\n'
    for sn,item in enumerate(recall_knowledge):
        displaydata = { "doc": item['doc'],"score": item['score']}
        doc_category  = item['doc_classify']
        doc_title =  item['doc_title']
        text += f'Doc[{sn+1}]:["{doc_title}"]-["{doc_category}"]\n{json.dumps(displaydata,ensure_ascii=False)}\n'
    text += '\n```'
    return text

def main_entry_new(user_id:str,wsconnection_id:str,session_id:str, query_input:str, embedding_model_endpoint:str, llm_model_endpoint:str, llm_model_name:str, aos_endpoint:str, aos_index:str, aos_knn_field:str, aos_result_num:int, kendra_index_id:str, 
                   kendra_result_num:int,use_qa:bool,wsclient=None,msgid:str='',max_tokens:int = 2048,temperature:float = 0.1,template:str = '',imgurl:str = None,multi_rounds:bool = False, hide_ref:bool = False,use_stream:bool=False):
    """
    Entry point for the Lambda function.

    Parameters:
        session_id (str): The ID of the session.
        query_input (str): The query input.
        embedding_model_endpoint (str): The endpoint of the embedding model.
        llm_model_endpoint (str): The endpoint of the language model.
        aos_endpoint (str): The endpoint of the AOS engine.
        aos_index (str): The index of the AOS engine.
        aos_knn_field (str): The knn field of the AOS engine.
        aos_result_num (int): The number of results of the AOS engine.
        kendra_index_id (str): The ID of the Kendra index.
        kendra_result_num (int): The number of results of the Kendra Service.

    return: answer(str)
    """
    # STOP=[f"\n{A_Role}", f"\n{B_Role}"]
    global STOP,TRACE_LOGGER
    #如果是reset命令，则清空历史聊天
    if query_input == RESET:
        delete_session(session_id,user_id)
        answer = '历史对话已清空'
        json_obj = {
            "query": query_input,
            "opensearch_doc":  [],
            "opensearch_knn_doc":  [],
            "kendra_doc": [],
            "knowledges" : [],
            "detect_query_type": '',
            "LLM_input": ''
        }
        json_obj['user_id'] = user_id
        json_obj['session_id'] = session_id
        json_obj['chatbot_answer'] = answer
        json_obj['conversations'] = []
        json_obj['timestamp'] = int(time.time())
        json_obj['log_type'] = "all"
        json_obj_str = json.dumps(json_obj, ensure_ascii=False)
        logger.info(json_obj_str)
        use_stream = False
        return answer,'',use_stream,'',[],[],[]
    
    logger.info("llm_model_name : {} ,use_stream :{}".format(llm_model_name,use_stream))
    llm = None
    stream_callback = CustomStreamingOutCallbackHandler(wsclient,msgid, wsconnection_id,llm_model_name,hide_ref,use_stream)
    if llm_model_name.startswith('claude'):
        # ACCESS_KEY, SECRET_KEY=get_bedrock_aksk()

        parameters = {
            "max_tokens_to_sample": max_tokens,
            "stop_sequences":STOP,
            "temperature":temperature,
            "top_p":0.95
        }

        model_id = BEDROCK_LLM_MODELID_LIST[llm_model_name] if llm_model_name == 'claude-instant' else BEDROCK_LLM_MODELID_LIST['claude-v2']

        llm = Bedrock(model_id=model_id, 
                      client=boto3_bedrock,
                      streaming=use_stream,
                      callbacks=[stream_callback],
                        model_kwargs=parameters)

    elif llm_model_name.startswith('gpt-3.5-turbo'):
        global openai_api_key
        llm=ChatOpenAI(model = llm_model_name,
                       openai_api_key = openai_api_key,
                       streaming = use_stream,
                       callbacks=[stream_callback],
                       temperature = temperature)
        
    elif use_stream:
        parameters = {
                "max_length": max_tokens,
                "temperature": temperature,
                "top_p":0.95
                }
        llmcontent_handler = SagemakerStreamContentHandler(
            callbacks=stream_callback
            )

        model_kwargs={'parameters':parameters,'history':[],'image':imgurl,'stream':use_stream}
        logging.info(f"model_kwargs:{model_kwargs}")
        llm = SagemakerStreamEndpoint(endpoint_name=llm_model_endpoint, 
                region_name=region, 
                model_kwargs=model_kwargs,
                content_handler=llmcontent_handler,
                endpoint_kwargs={'CustomAttributes':'accept_eula=true'} ##for llama2
                )
    else:
        parameters = {
            "max_length": max_tokens,
            "temperature": temperature,
            "top_p":0.95
        }

        model_kwargs={'parameters':parameters,'history':[],'image':imgurl}
        logging.info(f"model_kwargs:{model_kwargs}")
        llmcontent_handler = llmContentHandler()
        llm=SagemakerEndpoint(
                endpoint_name=llm_model_endpoint, 
                region_name=region, 
                model_kwargs=model_kwargs,
                content_handler=llmcontent_handler,
                endpoint_kwargs={'CustomAttributes':'accept_eula=true'} ##for llama2
            )
    
    
    # 1. get_session
    start1 = time.time()
    session_history = get_session(session_id=session_id)

    chat_coversions = [ (item[0],item[1]) for item in session_history]

    elpase_time = time.time() - start1
    logger.info(f'runing time of get_session : {elpase_time}s seconds')
    answer = None
    query_type = None
    # free_chat_coversions = []
    verbose = False
    logger.info(f'use QA: {use_qa}')
    final_prompt = ''
    origin_query = query_input
    intention = ''

    TRACE_LOGGER.trace(f'**Starting trace mode...**')
    if multi_rounds:
        before_rewrite = time.time()
        query_input = rewrite_query(origin_query, session_history, round_cnt=3)
        elpase_time_rewrite = time.time() - before_rewrite

        chat_history=''
        TRACE_LOGGER.trace(f'**Rewrite: {origin_query} => {query_input}, elpase_time:{elpase_time_rewrite}**')
        #add history parameter
        if isinstance(llm,SagemakerStreamEndpoint) or isinstance(llm,SagemakerEndpoint):
            chat_history=''
            llm.model_kwargs['history'] = chat_coversions[-2:]
        else:
            chat_history= get_chat_history(chat_coversions[-2:])
    else:
        chat_history=''


    
    doc_retriever = CustomDocRetriever.from_endpoints(embedding_model_endpoint=embedding_model_endpoint,
                                    aos_endpoint= aos_endpoint,
                                    aos_index=aos_index)
    
    TRACE_LOGGER.trace(f'**Using LLM model : {llm_model_name}**')
    cache_answer = None
    if use_qa:
        before_prefetch = time.time()
        TRACE_LOGGER.trace(f'**Prefetching cache...**')
        cache_repsonses = doc_retriever.knn_quick_prefetch(query_input)
        elpase_time_cache = time.time() - before_prefetch
        TRACE_LOGGER.trace(f'**Running time of prefetching cache: {elpase_time_cache:.3f}s**')
        if cache_repsonses:
            last_cache = cache_repsonses[-1]['doc']
            cache_answer = last_cache.split('\nAnswer:')[1]
            TRACE_LOGGER.trace(f"**Found caches:**")
            for sn,item in enumerate(cache_repsonses[::-1]):
                TRACE_LOGGER.trace(f"**[{sn+1}] [{item['doc_title']}] [{item['doc_type']}] [{item['doc_classify']}] [{item['score']:.3f}] author:[{item['doc_author']}]**")
                TRACE_LOGGER.trace(f"{item['doc']}")
        else:
            TRACE_LOGGER.trace(f"**No cache found**")

    detection = {'func': 'QA'}
    intention = detection['func']
    global INTENTION_LIST
    other_intentions = INTENTION_LIST.split(',')

    ##如果使用QA，且没有cache answer再需要进一步意图判断
    if use_qa and not cache_answer and len(other_intentions) > 0 and len(other_intentions[0]) > 1:
        before_detect = time.time()
        TRACE_LOGGER.trace(f'**Detecting intention...**')
        detection = detect_intention(query_input, fewshot_cnt=5)
        intention = detection['func']
        elpase_time_detect = time.time() - before_detect
        logger.info(f'detection: {detection}')
        logger.info(f'running time of detecting : {elpase_time_detect:.3f}s')
        TRACE_LOGGER.trace(f'**Running time of detecting: {elpase_time_detect:.3f}s**')
        TRACE_LOGGER.trace(f'**Detected intention: {intention}**')
    
    if not use_qa:
        intention = 'chat'

    if cache_answer:
        TRACE_LOGGER.trace('**Use Cache answer:**')
        reply_stratgy = ReplyStratgy.OTHER
        answer = cache_answer
        if use_stream:
            TRACE_LOGGER.postMessage(cache_answer)
        recall_knowledge,opensearch_knn_respose,opensearch_query_response = [],[],[]

    elif intention in ['chat', 'assist']:##如果不使用QA
        TRACE_LOGGER.trace(f'**Using Non-RAG {intention}...**')
        TRACE_LOGGER.trace('**Answer:**')
        reply_stratgy = ReplyStratgy.LLM_ONLY
        prompt_template = None
        answer = ''
        if intention == 'chat':
            prompt_template = create_chat_prompt_templete(llm_model_name=llm_model_name)
            llmchain = LLMChain(llm=llm,verbose=verbose,prompt =prompt_template )
            answer = llmchain.run({'question':query_input,'chat_history':chat_history,'role_bot':B_Role})
            final_prompt = prompt_template.format(question=query_input,role_bot=B_Role,chat_history=chat_history)
        elif intention == 'assist':
            prompt_template = create_assist_prompt_templete(llm_model_name=llm_model_name)
            llmchain = LLMChain(llm=llm,verbose=verbose,prompt =prompt_template )
            answer = llmchain.run({'question':query_input,'chat_history':chat_history,'role_bot':B_Role})
            final_prompt = prompt_template.format(question=query_input, role_bot=B_Role,chat_history=chat_history)

        answer = answer.replace('</response>','')
        recall_knowledge,opensearch_knn_respose,opensearch_query_response = [],[],[]

    elif intention == 'QA': ##如果使用QA
        # 2. aos retriever
        TRACE_LOGGER.trace('**Using RAG Chat...**')
        

        # doc_retriever = CustomDocRetriever.from_endpoints(embedding_model_endpoint=embedding_model_endpoint,
        #                             aos_endpoint= aos_endpoint,
        #                             aos_index=aos_index)
        # 3. check is it keyword search
        # exactly_match_result = aos_search(aos_endpoint, aos_index, "doc", query_input, exactly_match=True)
        ## 精准匹配对paragraph类型文档不太适用，先屏蔽掉 
        exactly_match_result = None

        start = time.time()
        ## 加上一轮的问题拼接来召回内容
        # query_with_history= get_question_history(chat_coversions[-2:])+query_input
        recall_knowledge = None
        opensearch_knn_respose = []
        opensearch_query_response = []
        if KNOWLEDGE_BASE_ID:
            TRACE_LOGGER.trace('**Retrieving knowledge from bedrock knowledgebase...**')
            recall_knowledge = doc_retriever.get_relevant_documents_from_bedrock(KNOWLEDGE_BASE_ID, query_input)
        else:
            TRACE_LOGGER.trace('**Retrieving knowledge from OpenSearch...**')
            recall_knowledge, opensearch_knn_respose, opensearch_query_response = doc_retriever.get_relevant_documents_custom(query_input) 

        elpase_time = time.time() - start
        logger.info(f'running time of opensearch_query : {elpase_time:.3f}s seconds')
        TRACE_LOGGER.trace(f'**Running time of retrieving knowledge : {elpase_time:.3f}s**')
        TRACE_LOGGER.trace(f'**Retrieved {len(recall_knowledge)} knowledge:**')
        TRACE_LOGGER.add_ref(f'\n\n**Refer to {len(recall_knowledge)} knowledge:**')

        ##添加召回文档到refdoc和tracelog, 按score倒序展示
        for sn,item in enumerate(recall_knowledge[::-1]):
            TRACE_LOGGER.trace(f"**[{sn+1}] [{item['doc_title']}] [{item['doc_classify']}] [{item['score']:.3f}] [{item['rank_score']:.3f}] author:[{item['doc_author']}]**")
            TRACE_LOGGER.trace(f"{item['doc']}")
            TRACE_LOGGER.add_ref(f"**[{sn+1}] [{item['doc_title']}] [{item['doc_classify']}] [{item['score']:.3f}] [{item['rank_score']:.3f}] author:[{item['doc_author']}]**")
            TRACE_LOGGER.add_ref(f"{item['doc']}")
        TRACE_LOGGER.trace('**Answer:**')

        def get_reply_stratgy(recall_knowledge):
            if not recall_knowledge:
                stratgy = ReplyStratgy.SAY_DONT_KNOW
                return stratgy

            global BM25_QD_THRESHOLD_HARD_REFUSE, BM25_QD_THRESHOLD_SOFT_REFUSE
            global KNN_QQ_THRESHOLD_HARD_REFUSE, KNN_QQ_THRESHOLD_SOFT_REFUSE
            global KNN_QD_THRESHOLD_HARD_REFUSE, KNN_QD_THRESHOLD_SOFT_REFUSE

            stratgy = ReplyStratgy.RETURN_OPTIONS
            for item in recall_knowledge:
                if item['score'] > 1.0:
                    if item['score'] > BM25_QD_THRESHOLD_SOFT_REFUSE:
                        stratgy = ReplyStratgy.WITH_LLM
                    elif item['score'] > BM25_QD_THRESHOLD_HARD_REFUSE:
                        stratgy = ReplyStratgy(min(ReplyStratgy.HINT_LLM_REFUSE.value, stratgy.value))
                    else:
                        stratgy = ReplyStratgy(min(ReplyStratgy.RETURN_OPTIONS.value, stratgy.value))

                elif item['score'] <= 1.0:
                    if item['score'] > KNN_QD_THRESHOLD_SOFT_REFUSE:
                        stratgy = ReplyStratgy.WITH_LLM
                    elif item['score'] > KNN_QD_THRESHOLD_HARD_REFUSE:
                        stratgy = ReplyStratgy(min(ReplyStratgy.HINT_LLM_REFUSE.value, stratgy.value))
                    else:
                        stratgy = ReplyStratgy(min(ReplyStratgy.RETURN_OPTIONS.value, stratgy.value))
            return stratgy

        def choose_prompt_template(stratgy:Enum, template:str, llm_model_name:str):
            if stratgy == ReplyStratgy.WITH_LLM:
                return create_baichuan_prompt_template(template) if llm_model_name.startswith('baichuan') else create_qa_prompt_templete(template)
            elif stratgy == ReplyStratgy.HINT_LLM_REFUSE:
                return create_soft_refuse_template(template)
            else:
                raise RuntimeError(
                    "unsupported startgy..."
                )

        reply_stratgy = get_reply_stratgy(recall_knowledge)

        if exactly_match_result and recall_knowledge:
            answer = exactly_match_result[0]["doc"]
            hide_ref= True ## 隐藏ref doc
            if use_stream:
                TRACE_LOGGER.postMessage(answer)
        elif reply_stratgy == ReplyStratgy.RETURN_OPTIONS:
            some_reference, multi_choice_field = format_knowledges(recall_knowledge[::2])
            answer = f"我不太确定，这有两条可能相关的信息，供参考：\n=====\n{some_reference}\n====="
            hide_ref= True ## 隐藏ref doc
            if use_stream:
                TRACE_LOGGER.postMessage(answer)
        elif reply_stratgy == ReplyStratgy.SAY_DONT_KNOW:
            answer = "我不太清楚，问问人工吧。"
            hide_ref= True ## 隐藏ref doc
            if use_stream:
                TRACE_LOGGER.postMessage(answer)
        else:      
            prompt_template = choose_prompt_template(reply_stratgy, template, llm_model_name)
            llmchain = LLMChain(llm=llm,verbose=verbose,prompt =prompt_template )

            # context = "\n".join([doc['doc'] for doc in recall_knowledge])
            context, multi_choice_field = format_knowledges(recall_knowledge)
            ask_user_prompts = [ f"- If you are not sure about which {field} user ask for, please ask user to clarify it, don't say anything else." for field in multi_choice_field ]
            ask_user_prompts_str = "\n".join(ask_user_prompts)

            chat_history = '' ##QA 场景下先不使用history
            ##最终的answer
            try:
                answer = llmchain.run({'question':query_input,'context':context,'chat_history':chat_history,'role_bot':B_Role, 'ask_user_prompt':ask_user_prompts_str })
            except Exception as e:
                answer = str(e)
            ##最终的prompt日志
            final_prompt = prompt_template.format(question=query_input,role_bot=B_Role,context=context,chat_history=chat_history,ask_user_prompt=ask_user_prompts_str)
            # print(final_prompt)
            # print(answer)
    else:
        #call agent for other intentions
        TRACE_LOGGER.trace('**Using Agent...**')
        reply_stratgy = ReplyStratgy.AGENT
        use_bedrock = "False"
        if llm_model_name.startswith('claude'):
            use_bedrock = "True"
        answer,ref_doc = chat_agent(query_input, detection, use_bedrock=use_bedrock)
        recall_knowledge,opensearch_knn_respose,opensearch_query_response = [ref_doc],[],[]
        TRACE_LOGGER.add_ref(f'\n\n**Refer to {len(recall_knowledge)} knowledge:**')
        TRACE_LOGGER.add_ref(f"**[1]** {ref_doc}")
        TRACE_LOGGER.trace(f'**Function call result:**\n\n{ref_doc}')
        TRACE_LOGGER.trace('**Answer:**')
        if use_stream:
            TRACE_LOGGER.postMessage(answer)

    answer = enforce_stop_tokens(answer, STOP)
    pattern = r'^根据[^，,]*[,|，]'
    answer = re.sub(pattern, "", answer)
    ref_text = ''
    # if not use_stream and recall_knowledge and hide_ref == False:
        # ref_text = format_reference(recall_knowledge)
    ref_text = TRACE_LOGGER.dump_refs(use_stream)

    json_obj = {
        "query": query_input,
        "origin_query" : origin_query,
        "intention" : intention,
        "opensearch_doc":  opensearch_query_response,
        "opensearch_knn_doc":  opensearch_knn_respose,
        "kendra_doc": [],
        "knowledges" : recall_knowledge,
        "LLM_input": final_prompt,
        "LLM_model_name": llm_model_name,
        "reply_stratgy" : reply_stratgy.name
    }
    json_obj['user_id'] = user_id
    json_obj['session_id'] = session_id
    json_obj['msgid'] = msgid
    json_obj['chatbot_answer'] = answer
    json_obj['ref_docs'] = ref_text
    json_obj['conversations'] = chat_coversions[-1:]
    json_obj['timestamp'] = int(time.time())
    json_obj['log_type'] = "all"
    json_obj_str = json.dumps(json_obj, ensure_ascii=False)
    logger.info(json_obj_str)

    start = time.time()
    if session_id != 'OnlyForDEBUG':
        update_session(session_id=session_id, question=query_input, answer=answer, intention=intention,msgid=msgid)
    elpase_time = time.time() - start
    elpase_time1 = time.time() - start1
    logger.info(f'runing time of update_session : {elpase_time}s seconds')
    logger.info(f'runing time of all  : {elpase_time1}s seconds')
    return answer,ref_text,use_stream,query_input,opensearch_query_response,opensearch_knn_respose,recall_knowledge

def delete_doc_index(obj_key,embedding_model,index_name):
    def delete_aos_index(obj_key,index_name,size=50):
        aos_endpoint = os.environ.get("aos_endpoint", "")
        client = OpenSearch(
                    hosts=[{'host':aos_endpoint, 'port': 443}],
                    http_auth = awsauth,
                    use_ssl=True,
                    verify_certs=True,
                    connection_class=RequestsHttpConnection
                )
        query =  {
                "size":size,
                "query" : {
                    "match_phrase":{
                        "doc_title": obj_key
                    }
                }
            }
        response = client.search(
            body=query,
            index=index_name
        )
        doc_ids = [hit["_id"] for hit in response["hits"]["hits"]]
        should_continue = False
        for doc_id in doc_ids:
            should_continue = True
            try:
                client.delete(index=index_name, id=doc_id)
                logger.info(f"delete:{doc_id}")
            except Exception as e:
                logger.info(f"delete:{doc_id}")
                continue

        return should_continue
    
    ##删除ddb里的索引
    dynamodb = boto3.client('dynamodb')
    try:
        dynamodb.delete_item(
            TableName=DOC_INDEX_TABLE,
            Key={
                'filename': {'S': obj_key},
                'embedding_model': {'S': embedding_model}
            }
        )
    except Exception as e:
        logger.info(str(e))

    ##删除aos里的索引
    should_continue = True
    while should_continue:
        should_continue = delete_aos_index(obj_key,index_name)
    
def list_doc_index ():
    dynamodb = boto3.client('dynamodb')
    scan_params = {
        'TableName': DOC_INDEX_TABLE,
        'Select': 'ALL_ATTRIBUTES',  # Return all attributes
    }
    try:
        response = dynamodb.scan(**scan_params)
        return response['Items']
    except Exception as e:
        logger.info(str(e))
        return []

def get_template(id):
    dynamodb = boto3.client('dynamodb')
    if id:
        params = {
            'TableName': os.environ.get('prompt_template_table'),
            'Key': {'id': {'S': id}},  # Return all attributes
        }
        try:
            response = dynamodb.get_item(**params)
            return response['Item']
        except Exception as e:
            logger.info(str(e))
            return None   
    else:
        params = {
            'TableName': os.environ.get('prompt_template_table'),
            'Select': 'ALL_ATTRIBUTES',  # Return all attributes
        }
        try:
            response = dynamodb.scan(**params)
            return response['Items']
        except Exception as e:
            logger.info(str(e))
            return []
    
def add_template(item):
    dynamodb = boto3.client('dynamodb')
    params = {
        'TableName': os.environ.get('prompt_template_table'),
        'Item': item,  
    }
    try:
        dynamodb.put_item(**params)
        return True
    except Exception as e:
        logger.info(str(e))
        return False

def delete_template(key):
    dynamodb = boto3.client('dynamodb')
    params = {
        'TableName': os.environ.get('prompt_template_table'),
        'Key': key,  
    }
    try:
        dynamodb.delete_item(**params)
        return True
    except Exception as e:
        logger.info(str(e))
        return False

def generate_s3_image_url(bucket_name, key, expiration=3600):
    s3_client = boto3.client('s3')
    url = s3_client.generate_presigned_url(
        'get_object',
         Params={'Bucket': bucket_name, 'Key': key},
         ExpiresIn=expiration
    )
    return url


## 1. write the feedback in logs and loaded to kinesis
## 2. if lambda_feedback is setup, then call lambda_feedback for other managment operations
def handle_feedback(event):
    method = event.get('method')
    
    ##invoke feedback lambda to store in ddb
    fn = os.environ.get('lambda_feedback')
    if method == 'post':
        results = True
        body = event.get('body')
        ## actions types: thumbs-up,thumbs-down,cancel-thumbs-up,cancel-thumbs-down
        timestamp = time.time()
        utc_datetime = datetime.utcfromtimestamp(timestamp)
        # Set the timezone to UTC+8
        utc8_timezone = pytz.timezone('Asia/Shanghai')
        datetime_utc8 = utc_datetime.replace(tzinfo=pytz.utc).astimezone(utc8_timezone)
        json_obj = {
                "opensearch_doc":  [], #for kiness firehose log subscription filter name
                "log_type":'feedback',
                "msgid":body.get('msgid'),
                "timestamp":str(datetime_utc8),
                "username":body.get('username'),
                "session_id":body.get('session_id'),
                "action":body.get('action'),
                "feedback":body.get('feedback')
            }
        json_obj_str = json.dumps(json_obj, ensure_ascii=False)
        logger.info(json_obj_str)

        json_obj = {**json_obj,**body,'method':method}
        if fn:
            response = lambda_client.invoke(
                    FunctionName = fn,
                    InvocationType='RequestResponse',
                    Payload=json.dumps(json_obj)
                )
            payload_json = json.loads(response.get('Payload').read())
            logger.info(payload_json)
            results = payload_json['body']
            if response['StatusCode'] != 200 or not results:
                logger.info(f"invoke lambda feedback StatusCode:{response['StatusCode']} and result {results}")
                results = False
        return results
    elif method == 'get':
        results = []
        body = event.get('body')
        json_obj = {**body,'method':method}
        if fn:
            response = lambda_client.invoke(
                    FunctionName = fn,
                    InvocationType='RequestResponse',
                    Payload=json.dumps(json_obj)
                )
            if response['StatusCode'] == 200:
                payload_json = json.loads(response.get('Payload').read())
                results = payload_json['body']
        return results   
    elif method == 'delete':  
        results = True
        body = event.get('body')
        json_obj = {**body,'method':method}
        if fn:
            response = lambda_client.invoke(
                    FunctionName = fn,
                    InvocationType='RequestResponse',
                    Payload=json.dumps(json_obj)
                )
            if response['StatusCode'] == 200:
                payload_json = json.loads(response.get('Payload').read())
                results = payload_json['body']
        return results  


@handle_error
def lambda_handler(event, context):
    # "model": 模型的名称
    # "chat_name": 对话标识，后端用来存储查找实现多轮对话 session
    # "prompt": 用户输入的问题
    # "max_tokens": 2048
    # "temperature": 0.9
    logger.info(f"event:{event}")
    method = event.get('method')
    resource = event.get('resource')
    global CHANNEL_RET_CNT
    CHANNEL_RET_CNT = event.get('channel_cnt', 10)
    logger.info(f'channel_cnt:{CHANNEL_RET_CNT}')

    ###其他管理操作 start
    ##如果是删除doc index的操作
    if method == 'delete' and resource == 'docs':
        logger.info(f"delete doc index of:{event.get('filename')}/{event.get('embedding_model')}/{event.get('index_name')}")
        delete_doc_index(event.get('filename'),event.get('embedding_model'),event.get('index_name'))
        return {'statusCode': 200}
    ## 如果是get doc index操作
    if method == 'get' and resource == 'docs':
        results = list_doc_index()
        return {'statusCode': 200,'body':results }
    ## 如果是get template 操作
    if method == 'get' and resource == 'template':
        id = event.get('id')
        results = get_template(id)
        return {'statusCode': 200,'body': {} if results is None else results }
    ## 如果是add a template 操作
    if method == 'post' and resource == 'template':
        body = event.get('body')
        item = {
            'id': {'S': body.get('id')},
            'template_name':{'S':body.get('template_name','')},
            'template':{'S':body.get('template','')},
            'comment':{'S':body.get('comment','')},
            'username':{'S':body.get('username','')}
        }
        result = add_template(item)
        return {'statusCode': 200 if result else 500,'body':result }
     ## 如果是delete a template 操作
    if method == 'delete' and resource == 'template':
        body = event.get('body')
        key = {
            'id': {'S': body.get('id')}
        }
        result = delete_template(key)
        return {'statusCode': 200 if result else 500,'body':result }

    ## 处理feedback action
    if method in ['post','get','delete'] and resource == 'feedback':
        results = handle_feedback(event)
        return {'statusCode': 200 if results else 500,'body':results}


    ####其他管理操作 end

    # input_json = json.loads(event['body'])
    ws_endpoint = event.get('ws_endpoint')
    if ws_endpoint:
        wsclient = boto3.client('apigatewaymanagementapi', endpoint_url=ws_endpoint)
    else:
        wsclient = None
    global openai_api_key
    openai_api_key = event.get('OPENAI_API_KEY') 
    hide_ref = event.get('hide_ref',False)
    retrieve_only = event.get('retrieve_only',False)
    session_id = event['chat_name']
    wsconnection_id = event.get('wsconnection_id',session_id)
    question = event['prompt']
    model_name = event['model'] if event.get('model') else event.get('model_name','')
    embedding_endpoint = event.get('embedding_model',os.environ.get("embedding_endpoint")) 
    use_qa = event.get('use_qa',False)
    multi_rounds = event.get('multi_rounds',False)
    use_stream = event.get('use_stream',False)
    user_id = event.get('user_id','')
    use_trace = event.get('use_trace',True)
    template_id = event.get('template_id')
    msgid = event.get('msgid')
    max_tokens = event.get('max_tokens',2048)
    temperature =  event.get('temperature',0.1)
    imgurl = event.get('imgurl')
    image_path = ''
    if imgurl:
        if imgurl.startswith('https://'):
            image_path = imgurl
        else:
            bucket,imgobj = imgurl.split('/',1)
            image_path = generate_s3_image_url(bucket,imgobj)
        logger.info(f"image_path:{image_path}")

    ## 用于trulength接口，只返回recall 知识
    if retrieve_only:
        doc_retriever = CustomDocRetriever.from_endpoints(embedding_model_endpoint=embedding_endpoint,
                                    aos_endpoint= os.environ.get("aos_endpoint", ""),
                                    aos_index=os.environ.get("aos_index", ""))
        recall_knowledge,opensearch_knn_respose,opensearch_query_response = doc_retriever.get_relevant_documents_custom(question) 
        extra_info = {"query_input": question, "opensearch_query_response" : opensearch_query_response, "opensearch_knn_respose": opensearch_knn_respose,"recall_knowledge":recall_knowledge }
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': [{"id": str(uuid.uuid4()), "extra_info" : extra_info,} ]
        }
        
    ##获取前端给的系统设定，如果没有，则使用lambda里的默认值
    global B_Role,SYSTEM_ROLE_PROMPT
    B_Role = event.get('system_role',B_Role)
    SYSTEM_ROLE_PROMPT = event.get('system_role_prompt',SYSTEM_ROLE_PROMPT)
    
    logger.info(f'system_role:{B_Role},system_role_prompt:{SYSTEM_ROLE_PROMPT}')

    llm_endpoint = os.environ.get('llm_model_endpoint')


    # 获取当前时间戳
    request_timestamp = time.time()  # 或者使用 time.time_ns() 获取纳秒级别的时间戳
    logger.info(f'request_timestamp :{request_timestamp}')
    logger.info(f"event:{event}")
    # logger.info(f"context:{context}")

    # 创建日志组和日志流
    log_group_name = '/aws/lambda/{}'.format(context.function_name)
    log_stream_name = context.aws_request_id
    client = boto3.client('logs')
    # 接收触发AWS Lambda函数的事件
    logger.info('The main brain has been activated, aws🚀!')

    # 1. 获取环境变量

    # embedding_endpoint = os.environ.get("embedding_endpoint", "")
    aos_endpoint = os.environ.get("aos_endpoint", "")
    aos_index = os.environ.get("aos_index", "")
    aos_knn_field = os.environ.get("aos_knn_field", "")
    aos_result_num = int(os.environ.get("aos_results", 4))

    Kendra_index_id = os.environ.get("Kendra_index_id", "")
    Kendra_result_num = int(os.environ.get("Kendra_result_num", 0))
    # Opensearch_result_num = int(os.environ.get("Opensearch_result_num", ""))
    prompt_template = ''

    ##如果指定了prompt 模板
    if template_id and template_id != 'default':
        prompt_template = get_template(template_id)
        prompt_template = '' if prompt_template is None else prompt_template['template']['S']
    logger.info(f'user_id : {user_id}')
    logger.info(f'prompt_template_id : {template_id}')
    logger.info(f'prompt_template : {prompt_template}')
    logger.info(f'model_name : {model_name}')
    logger.info(f'llm_endpoint : {llm_endpoint}')
    logger.info(f'embedding_endpoint : {embedding_endpoint}')
    logger.info(f'aos_endpoint : {aos_endpoint}')
    logger.info(f'aos_index : {aos_index}')
    logger.info(f'aos_knn_field : {aos_knn_field}')
    logger.info(f'aos_result_num : {aos_result_num}')
    logger.info(f'Kendra_index_id : {Kendra_index_id}')
    logger.info(f'Kendra_result_num : {Kendra_result_num}')
    logger.info(f'use multiple rounds: {multi_rounds}')
    logger.info(f'intention list: {INTENTION_LIST}')
    global TRACE_LOGGER
    TRACE_LOGGER = TraceLogger(wsclient=wsclient,msgid=msgid,connectionId=wsconnection_id,stream=use_stream,use_trace=use_trace,hide_ref=hide_ref)

    main_entry_start = time.time()  # 或者使用 time.time_ns() 获取纳秒级别的时间戳
    answer,ref_text,use_stream,query_input,opensearch_query_response,opensearch_knn_respose,recall_knowledge = main_entry_new(user_id,wsconnection_id,session_id, question, embedding_endpoint, llm_endpoint, model_name, aos_endpoint, aos_index, aos_knn_field, aos_result_num,
                       Kendra_index_id, Kendra_result_num,use_qa,wsclient,msgid,max_tokens,temperature,prompt_template,image_path,multi_rounds,hide_ref,use_stream)
    main_entry_elpase = time.time() - main_entry_start  # 或者使用 time.time_ns() 获取纳秒级别的时间戳
    logger.info(f'runing time of main_entry : {main_entry_elpase}s seconds')
    if use_stream: ##只有当stream输出时，把这条trace放到最后一个chunk
        TRACE_LOGGER.trace(f'\n\n**Total running time : {main_entry_elpase:.3f}s**')
    if TRACE_LOGGER.use_trace:
        tracelogs_str = TRACE_LOGGER.dump_logs_to_string()
        ## 返回非stream的结果，把这条trace放到末尾
        answer =f'{tracelogs_str}\n\n{answer}\n\n**Total running time : {main_entry_elpase:.3f}s**'
    else:
        answer = answer if hide_ref else f'{answer}{ref_text}'
    # 2. return rusult

    # 处理

    # Response:
    # "id": "设置一个uuid"
    # "created": "1681891998"
    # "model": "模型名称"
    # "choices": [{"text": "模型回答的内容"}]
    # "usage": {"prompt_tokens": 58, "completion_tokens": 15, "total_tokens": 73}}]
    extra_info = {}
    if session_id == 'OnlyForDEBUG':
        extra_info = {"query_input": query_input, "opensearch_query_response" : opensearch_query_response, "opensearch_knn_respose": opensearch_knn_respose, "recall_knowledge":recall_knowledge }
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': [{"id": str(uuid.uuid4()),
                  "use_stream":use_stream,
                             "created": request_timestamp,
                             "useTime": time.time() - request_timestamp,
                             "model": "main_brain",
                             "choices": [{"text": answer}],
                            #  [{"text": "{}[{}]".format(answer, model_name)}],
                             "extra_info" : extra_info,
                             "usage": {"prompt_tokens": 58, "completion_tokens": 15, "total_tokens": 73}},
                            # {"id": uuid.uuid4(),
                            #  "created": request_timestamp,
                            #  "useTime": int(time.time()) - request_timestamp,
                            #  "model": "模型名称",
                            #  "choices":
                            #  [{"text": "2 模型回答的内容"}],
                            #  "usage": {"prompt_tokens": 58, "completion_tokens": 15, "total_tokens": 73}}
                            ]
    }