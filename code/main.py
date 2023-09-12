import json
import logging
import time
import os
import re
from botocore import config
from botocore.exceptions import ClientError
from datetime import datetime, timedelta
import boto3
import time
import requests
import uuid
# from transformers import AutoTokenizer
from enum import Enum
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain import PromptTemplate, SagemakerEndpoint
from langchain.chains import LLMChain,ConversationalRetrievalChain,ConversationChain
from langchain.schema import BaseRetriever
from langchain.schema import Document
from langchain.llms.bedrock import Bedrock
from pydantic import BaseModel,Extra,root_validator
import openai
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, Dict, List, Union,Mapping, Optional, TypeVar, Union
from langchain.schema import LLMResult
from langchain.llms.base import LLM
import io
import math


credentials = boto3.Session().get_credentials()
region = boto3.Session().region_name
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, 'es', session_token=credentials.token)

DOC_INDEX_TABLE= 'chatbot_doc_index'
logger = logging.getLogger()
logger.setLevel(logging.INFO)
sm_client = boto3.client("sagemaker-runtime")
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
STOP=[f"\n{A_Role_en}", f"\n{A_Role}", f"\n{Fewshot_prefix_Q}"]

KNN_THRESHOLD = float(os.environ.get('knn_threshold',0.5))
TOP_K = int(os.environ.get('TOP_K',4))
INVERTED_HRESHOLD =float(os.environ.get('inverted_theshold',10.0))
NEIGHBORS = int(os.environ.get('neighbors',1))


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
    def __init__(self,wsclient:str,msgid:str,connectionId:str ,model_name:str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.wsclient = wsclient
        self.connectionId = connectionId
        self.msgid = msgid
        self.model_name= model_name,
        self.recall_knowledge = None

    def add_recall_knowledge(self,recall_knowledge):

        self.recall_knowledge = recall_knowledge

    def postMessage(self,data):
        try:
            self.wsclient.post_to_connection(Data = data.encode('utf-8'),  ConnectionId=self.connectionId)
        except Exception as e:
            print (f'post {json.dumps(data)} to_wsconnection error:{str(e)}')

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
        data = json.dumps({ 'msgid':self.msgid, 'role': "AI", 'text': {'content':token} })
        self.postMessage(data)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        data = json.dumps({ 'msgid':self.msgid, 'role': "AI", 'text': {'content':f' [{self.model_name}]' } })
        self.postMessage(data)
        if self.recall_knowledge:
            text = format_reference(self.recall_knowledge)
            data = json.dumps({ 'msgid':self.msgid, 'role': "AI", 'text': {'content':f'{text}```'} })
            self.postMessage(data)
        

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        data = json.dumps({ 'msgid':self.msgid, 'role': "AI", 'text': {'content':str(error[0])+'[DONE]'} })
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

class CustomDocRetriever(BaseRetriever,BaseModel):
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
        recall_knowledge,_,_= self.get_relevant_documents_custom(query_input)
        top_k_results = []
        for item in recall_knowledge:
            top_k_results.append(Document(page_content=item.get('doc')))
        return top_k_results
       

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError

    
    def get_relevant_documents_custom(self, query_input: str):
        start = time.time()
        query_embedding = get_vector_by_sm_endpoint(query_input, sm_client, self.embedding_model_endpoint)
        aos_client = OpenSearch(
                hosts=[{'host': self.aos_endpoint, 'port': 443}],
                http_auth = awsauth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection
            )
        opensearch_knn_respose = search_using_aos_knn(aos_client,query_embedding[0], self.aos_index)
        elpase_time = time.time() - start
        logger.info(f'runing time of opensearch_knn : {elpase_time}s seconds')
        
        # 4. get AOS invertedIndex recall
        start = time.time()
        opensearch_query_response = aos_search(aos_client, self.aos_index, "doc", query_input)
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
                    if item['id'] not in unique_ids:
                        opensearch_knn_nodup.append((item['doc'], item['score'],item['idx'], item['doc_title'], item['id'],item['doc_category'],item['doc_type']))
                        unique_ids.add(item['id'])
                
                opensearch_bm25_nodup = []
                unique_ids = set()
                for item in opensearch_query_response:
                    if item['id'] not in unique_ids:
                        opensearch_bm25_nodup.append((item['doc'], item['score'], item['idx'], item['doc_title'],item['id'],item['doc_category'],item['doc_type']))
                        unique_ids.add(item['id'])

                opensearch_knn_nodup.sort(key=lambda x: x[1])
                opensearch_bm25_nodup.sort(key=lambda x: x[1])
                
                half_topk = math.ceil(topk/2) 
    
                kg_combine_result = [ { "doc": item[0], "score": item[1],"idx":item[2],"doc_title":item[3], "doc_category":item[5],"doc_type": item[6]  } for item in opensearch_knn_nodup[-1*half_topk:]]
                knn_kept_doc = [ item[4] for item in opensearch_knn_nodup[-1*half_topk:] ]

                bm25_count = 0
                for item in opensearch_bm25_nodup[::-1]:
                    if item[4] not in knn_kept_doc:
                        kg_combine_result.append({ "doc": item[0], "score": item[1],"idx":item[2],"doc_title":item[3],"doc_category":item[5],"doc_type": item[6] })
                        bm25_count += 1
                    if bm25_count+len(knn_kept_doc) >= topk:
                        break
                ##继续填补不足的召回
                step_knn = 0
                step_bm25 = 0
                while topk - len(kg_combine_result)>0:
                    if len(opensearch_knn_nodup) > half_topk:
                        kg_combine_result += [{ "doc": item[0], "score": item[1],"idx":item[2],"doc_title":item[3], "doc_category":item[5],"doc_type": item[6]  } for item in opensearch_knn_nodup[-1*half_topk-1-step_knn:-1*half_topk-step_knn]]
                        step_knn += 1
                    elif len(opensearch_bm25_nodup) > half_topk:
                        kg_combine_result += [{ "doc": item[0], "score": item[1],"idx":item[2],"doc_title":item[3], "doc_category":item[5],"doc_type": item[6]  } for item in opensearch_bm25_nodup[-1*half_topk-1-step_bm25:-1*half_topk-step_bm25]]
                        step_bm25 += 1
                    else:
                        break

                return kg_combine_result

            knn_threshold = KNN_THRESHOLD
            inverted_theshold = INVERTED_HRESHOLD
            filter_knn_result = [ item for item in opensearch_knn_respose if item['score'] > knn_threshold ]
            filter_inverted_result = [ item for item in opensearch_query_response if item['score'] > inverted_theshold ]
            
            ret_content = get_topk_items(filter_knn_result, filter_inverted_result, TOP_K)
            logger.info(f'get_topk_items:{len(ret_content)}')
            return ret_content
        
        recall_knowledge = combine_recalls(opensearch_knn_respose, opensearch_query_response)

        return recall_knowledge,opensearch_knn_respose,opensearch_query_response


class ErrorCode:
    DUPLICATED_INDEX_PREFIX = "DuplicatedIndexPrefix"
    DUPLICATED_WITH_INACTIVE_INDEX_PREFIX = "DuplicatedWithInactiveIndexPrefix"
    OVERLAP_INDEX_PREFIX = "OverlapIndexPrefix"
    OVERLAP_WITH_INACTIVE_INDEX_PREFIX = "OverlapWithInactiveIndexPrefix"
    INVALID_INDEX_MAPPING = "InvalidIndexMapping"


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

# kendra





def query_kendra(Kendra_index_id="", lang="zh", search_query_text="what is s3?", Kendra_result_num=3):
    # 连接到Kendra
    client = boto3.client('kendra')

    # 构造Kendra查询请求
    query_result = client.query(
        IndexId=Kendra_index_id,
        QueryText=search_query_text,
        AttributeFilter={
            "EqualsTo": {
                "Key": "_language_code",
                "Value": {
                    "StringValue": lang
                }
            }
        }
    )
    # print(query_result['ResponseMetadata']['HTTPHeaders'])
    # kendra_took = query_result['ResponseMetadata']['HTTPHeaders']['x-amz-time-millis']
    # 创建一个结果列表
    results = []

    # 将每个结果添加到结果列表中
    for result in query_result['ResultItems']:
        # 创建一个字典来保存每个结果
        result_dict = {}

        result_dict['score'] = 0.0
        result_dict['doc_type'] = "P"

        # 如果有可用的总结
        if 'DocumentExcerpt' in result:
            result_dict['doc'] = result['DocumentExcerpt']['Text']
        else:
            result_dict['doc'] = ''

        # 将结果添加到列表中
        results.append(result_dict)

    # 输出结果列表
    return results[:Kendra_result_num]


def is_chinese(string):
    for char in string:
        if '\u4e00' <= char <= '\u9fff':
            return True

    return False

# AOS
def get_vector_by_sm_endpoint(questions, sm_client, endpoint_name):
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
                "is_query" : True,
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
    opensearch_knn_respose = [{'idx':item['_source'].get('idx',1),'doc_category':item['_source']['doc_category'],'doc_title':item['_source']['doc_title'],'id':item['_id'],'doc':"{}{}{}".format(item['_source']['doc'], QA_SEP, item['_source']['content']),"doc_type":item["_source"]["doc_type"],"score":item["_score"]}  for item in query_response["hits"]["hits"]]
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
        result_arr = [ {'idx':item['_source'].get('idx',0),'doc_category':item['_source']['doc_category'],'doc_title':item['_source']['doc_title'],'id':item['_id'],'doc': item['_source']['content'], 'doc_type': item['_source']['doc_type'], 'score': item['_score']} for item in query_response["hits"]["hits"]]
    else:
        result_arr = [ {'idx':item['_source'].get('idx',0),'doc_category':item['_source']['doc_category'],'doc_title':item['_source']['doc_title'],'id':item['_id'],'doc':"{}{}{}".format(item['_source']['doc'], QA_SEP, item['_source']['content']), 'doc_type': item['_source']['doc_type'], 'score': item['_score']} for item in query_response["hits"]["hits"]]

    return result_arr

def delete_session(session_id):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(chat_session_table)
    try:
        table.delete_item(
        Key={
            'session-id': session_id,
        })
    except Exception as e:
        logger.info(f"delete session failed {str(e)}")

        
def get_session(session_id):

    table_name = chat_session_table
    dynamodb = boto3.resource('dynamodb')

    # table name
    table = dynamodb.Table(table_name)
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
def update_session(session_id, question, answer, intention):

    table_name = chat_session_table
    dynamodb = boto3.resource('dynamodb')

    # table name
    table = dynamodb.Table(table_name)
    operation_result = ""

    response = table.get_item(Key={'session-id': session_id})

    if "Item" in response.keys():
        # print("****** " + response["Item"]["content"])
        chat_history = json.loads(response["Item"]["content"])
    else:
        # print("****** No result")
        chat_history = []

    chat_history.append([question, answer, intention])
    content = json.dumps(chat_history)

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


class QueryType(Enum):
    KeywordQuery   = "KeywordQuery"       #用户仅仅输入了一些关键词（2 token)
    KnowledgeQuery = "KnowledgeQuery"     #用户输入的需要参考知识库有关来回答
    Conversation   = "Conversation"       #用户输入的是跟知识库无关的问题





def qa_knowledge_fewshot_build(recalls):
    ret_context = []
    for recall in recalls:
        if recall['doc_type'] == 'Question':
            q, a = recall['doc'].split(QA_SEP)
            qa_example = "{}: {}\n{}: {}".format(Fewshot_prefix_Q, q, Fewshot_prefix_A, a)
            ret_context.append(qa_example)
        elif recall['doc_type'] == 'Paragraph':
            ret_context.append(recall['doc'])

    context_str = "\n\n".join(ret_context)
    return context_str


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

def create_qa_prompt_templete(prompt_template):
    if prompt_template == '':
        prompt_template_zh = """{system_role_prompt} {role_bot}\n请根据反括号中的内容提取相关信息回答问题:\n```\n{chat_history}{context}\n```\n如果反括号中的内容为空,则回答不知道.\n用户:{question}"""
    else:
        prompt_template_zh = prompt_template
    PROMPT = PromptTemplate(
        template=prompt_template_zh,
        partial_variables={'system_role_prompt':SYSTEM_ROLE_PROMPT},
        input_variables=["context",'question','chat_history','role_bot']
    )
    return PROMPT

def create_chat_prompt_templete(prompt_template):
    if prompt_template == '':
        prompt_template_zh = """{system_role_prompt} {role_bot}\n {chat_history}\n\n用户: {question}"""
    else:
        prompt_template_zh = prompt_template.replace('{context}','') ##remove{context}
    PROMPT = PromptTemplate(
        template=prompt_template_zh, 
        partial_variables={'system_role_prompt':SYSTEM_ROLE_PROMPT},
        input_variables=['question','chat_history','role_bot']
    )
    return PROMPT

def get_bedrock_aksk(secret_name='chatbot_bedrock', region_name = "us-west-2"):
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
    text = '\n```json\n#Reference\n'
    for sn,item in enumerate(recall_knowledge):
        displaydata = { "doc": item['doc'],"score": item['score']}
        doc_category  = item['doc_category'] 
        doc_title =  item['doc_title']
        text += f'Doc[{sn+1}]:["{doc_title}"]-["{doc_category}"]\n{json.dumps(displaydata,ensure_ascii=False)}\n'
    return text

def main_entry_new(session_id:str, query_input:str, embedding_model_endpoint:str, llm_model_endpoint:str, llm_model_name:str, aos_endpoint:str, aos_index:str, aos_knn_field:str,
                    aos_result_num:int, kendra_index_id:str, kendra_result_num:int,use_qa:bool,wsclient=None,msgid:str='',max_tokens:int = 2048,temperature:float = 0.01,template:str = '',imgurl:str = None,multi_rounds:bool = False):
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
    global STOP
    use_stream = False
    #如果是reset命令，则清空历史聊天
    if query_input == RESET:
        delete_session(session_id)
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

        json_obj['session_id'] = session_id
        json_obj['chatbot_answer'] = answer
        json_obj['conversations'] = []
        json_obj['timestamp'] = int(time.time())
        json_obj['log_type'] = "all"
        json_obj_str = json.dumps(json_obj, ensure_ascii=False)
        logger.info(json_obj_str)
        return answer,use_stream
    
    logger.info("llm_model_name : {}".format(llm_model_name))
    llm = None
    stream_callback = CustomStreamingOutCallbackHandler(wsclient,msgid, session_id,llm_model_name)
    if llm_model_name == 'claude':
        ACCESS_KEY, SECRET_KEY=get_bedrock_aksk()

        boto3_bedrock = boto3.client(
            service_name="bedrock",
            region_name="us-east-1",
            endpoint_url="https://bedrock.us-east-1.amazonaws.com",
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY
        )

        parameters = {
            "max_tokens_to_sample": max_tokens,
            "stop_sequences":STOP,
            "temperature":temperature,
            "top_p":1
        }
        
        llm = Bedrock(model_id="anthropic.claude-v1", client=boto3_bedrock, model_kwargs=parameters)
        # print("llm is anthropic.claude-v1")
    elif llm_model_name.startswith('gpt-3.5-turbo'):
        use_stream = True
        global openai_api_key
        llm=ChatOpenAI(model = llm_model_name,
                       openai_api_key = openai_api_key,
                       streaming = True,
                       callbacks=[stream_callback],
                       temperature = temperature)
        
    elif llm_model_name.endswith('stream'):
        use_stream = True
        parameters = {
                "max_length": max_tokens,
                "temperature": temperature,
                "top_p":1
                }
        llmcontent_handler = SagemakerStreamContentHandler(
            callbacks=stream_callback
            )
        # model_kwargs={'parameters':parameters,'history':[]}
        # if imgurl:
        model_kwargs={'parameters':parameters,'history':[],'image':imgurl}
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
            "top_p":1
        }
        # model_kwargs={'parameters':parameters,'history':[]}
        # if imgurl:
        model_kwargs={'parameters':parameters,'history':[],'image':imgurl}
        llmcontent_handler = llmContentHandler()
        llm=SagemakerEndpoint(
                endpoint_name=llm_model_endpoint, 
                region_name=region, 
                model_kwargs=model_kwargs,
                content_handler=llmcontent_handler,
                endpoint_kwargs={'CustomAttributes':'accept_eula=true'} ##for llama2
            )
    
    # sm_client = boto3.client("sagemaker-runtime")
    
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
    if not use_qa:##如果不使用QA
        query_type = QueryType.Conversation
        # free_chat_coversions = [ (item[0],item[1]) for item in session_history if item[2] == str(query_type)]
        # chat_history= get_chat_history(free_chat_coversions[-2:])
        chat_coversions = [ (item[0],item[1]) for item in session_history]
        if multi_rounds:
            ##add history parameter
            if isinstance(llm,SagemakerStreamEndpoint) or isinstance(llm,SagemakerEndpoint):
                chat_history=''
                llm.model_kwargs['history'] = chat_coversions[-3:]
            else:
                chat_history= get_chat_history(chat_coversions[-3:])
        else:
            chat_history=''
        
        prompt_template = create_chat_prompt_templete(template)
        llmchain = LLMChain(llm=llm,verbose=verbose,prompt =prompt_template )
        ##最终的answer
        answer = llmchain.run({'question':query_input,'chat_history':chat_history,'role_bot':B_Role})
        ##最终的prompt日志
        final_prompt = prompt_template.format(question=query_input,role_bot=B_Role,chat_history=chat_history)
        recall_knowledge,opensearch_knn_respose,opensearch_query_response = [],[],[]
    else: ##如果使用QA
        # 2. aos retriever
        doc_retriever = CustomDocRetriever.from_endpoints(embedding_model_endpoint=embedding_model_endpoint,
                                    aos_endpoint= aos_endpoint,
                                    aos_index=aos_index)
        # 3. check is it keyword search
        # exactly_match_result = aos_search(aos_endpoint, aos_index, "doc", query_input, exactly_match=True)
        ## 精准匹配对paragraph类型文档不太适用，先屏蔽掉 
        exactly_match_result = None

        start = time.time()
        ## 加上一轮的问题拼接来召回内容
        # query_with_history= get_question_history(chat_coversions[-2:])+query_input
        recall_knowledge,opensearch_knn_respose,opensearch_query_response = doc_retriever.get_relevant_documents_custom(query_input) 
        elpase_time = time.time() - start
        logger.info(f'runing time of opensearch_query : {elpase_time}s seconds')
        
        ##add history parameter
        if multi_rounds:
            if isinstance(llm,SagemakerStreamEndpoint) or isinstance(llm,SagemakerEndpoint):
                chat_history=''
                llm.model_kwargs['history'] = chat_coversions[-3:]
            else:
                chat_history= get_chat_history(chat_coversions[-3:])
        else:
            chat_history=''
            
        if exactly_match_result and recall_knowledge: 
            query_type = QueryType.KeywordQuery
            answer = exactly_match_result[0]["doc"]
            final_prompt = ''
            use_stream = False ##如果是直接匹配则不需要走流
        else:      
            ##添加召回引用
            stream_callback.add_recall_knowledge(recall_knowledge)
            query_type = QueryType.KnowledgeQuery
            prompt_template = create_baichuan_prompt_template(template) if llm_model_name.startswith('baichuan') else create_qa_prompt_templete(template) 
            llmchain = LLMChain(llm=llm,verbose=verbose,prompt =prompt_template )
            # context = "\n".join([doc['doc'] for doc in recall_knowledge])
            context = qa_knowledge_fewshot_build(recall_knowledge)
            ##最终的answer
            answer = llmchain.run({'question':query_input,'context':context,'chat_history':chat_history,'role_bot':B_Role })
            ##最终的prompt日志
            final_prompt = prompt_template.format(question=query_input,role_bot=B_Role,context=context,chat_history=chat_history)
            # print(final_prompt)
            # print(answer)

    answer = enforce_stop_tokens(answer, STOP)

    if not use_stream and recall_knowledge:
        text = format_reference(recall_knowledge)
        answer+= f'{text}```'

    json_obj = {
        "query": query_input,
        "opensearch_doc":  opensearch_query_response,
        "opensearch_knn_doc":  opensearch_knn_respose,
        "kendra_doc": [],
        "knowledges" : recall_knowledge,
        "detect_query_type": str(query_type),
        "LLM_input": final_prompt,
        "LLM_model_name": llm_model_name
    }

    json_obj['session_id'] = session_id
    json_obj['chatbot_answer'] = answer
    json_obj['conversations'] = chat_coversions[-3:]
    json_obj['timestamp'] = int(time.time())
    json_obj['log_type'] = "all"
    json_obj_str = json.dumps(json_obj, ensure_ascii=False)
    logger.info(json_obj_str)

    start = time.time()
    update_session(session_id=session_id, question=query_input, answer=answer, intention=str(query_type))
    elpase_time = time.time() - start
    elpase_time1 = time.time() - start1
    logger.info(f'runing time of update_session : {elpase_time}s seconds')
    logger.info(f'runing time of all  : {elpase_time1}s seconds')

    return answer,use_stream

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
            return {}   
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
        return {'statusCode': 200,'body':results }
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
        return {'statusCode': 200 if result else 500,'body':results }
     ## 如果是delete a template 操作
    if method == 'delete' and resource == 'template':
        body = event.get('body')
        key = {
            'id': {'S': body.get('id')}
        }
        result = delete_template(key)
        return {'statusCode': 200 if result else 500,'body':results }

    # input_json = json.loads(event['body'])
    ws_endpoint = event.get('ws_endpoint')
    if ws_endpoint:
        wsclient = boto3.client('apigatewaymanagementapi', endpoint_url=ws_endpoint)
    else:
        wsclient = None
    global openai_api_key
    openai_api_key = event.get('OPENAI_API_KEY') 
    
    session_id = event['chat_name']
    question = event['prompt']
    model_name = event['model'] if event.get('model') else event.get('model_name','')
    embedding_endpoint = event.get('embedding_model',os.environ.get("embedding_endpoint")) 
    use_qa = event.get('use_qa',False)
    multi_rounds = event.get('multi_rounds',False)
    template_id = event.get('template_id')
    msgid = event.get('msgid')
    max_tokens = event.get('max_tokens',2048)
    temperature =  event.get('temperature',0.01)
    imgurl = event.get('imgurl')
    image_path = ''
    if imgurl:
        if imgurl.startswith('https://'):
            image_path = imgurl
        else:
            bucket,imgobj = imgurl.split('/',1)
            image_path = generate_s3_image_url(bucket,imgobj)
        logger.info(f"image_path:{image_path}")

    ##获取前端给的系统设定，如果没有，则使用lambda里的默认值
    global B_Role,SYSTEM_ROLE_PROMPT
    B_Role = event.get('system_role',B_Role)
    SYSTEM_ROLE_PROMPT = event.get('system_role_prompt',SYSTEM_ROLE_PROMPT)
    
    logger.info(f'system_role:{B_Role},system_role_prompt:{SYSTEM_ROLE_PROMPT}')

    llm_endpoint = None
    if model_name == 'chatglm':
        llm_endpoint = os.environ.get('llm_{}_endpoint'.format(model_name))
    elif model_name == 'bloomz': 
        llm_endpoint =  os.environ.get('llm_{}_endpoint'.format(model_name))
    elif model_name == 'llama':
        llm_endpoint =  os.environ.get('llm_{}_endpoint'.format(model_name))
        pass
    elif model_name == 'alpaca':
        llm_endpoint =  os.environ.get('llm_{}_endpoint'.format(model_name))
        pass
    elif model_name == 'chatglm-stream':
        llm_endpoint = os.environ.get('llm_chatglm_stream_endpoint')
    elif model_name == 'visualglm':
        llm_endpoint = os.environ.get('llm_visualglm_endpoint')
    elif model_name == 'visualglm-stream':
        llm_endpoint = os.environ.get('llm_visualglm_stream_endpoint')
    elif model_name == 'other-stream':
        llm_endpoint = os.environ.get('llm_other_stream_endpoint')
    else:
        llm_endpoint = os.environ.get('llm_default_endpoint')
        pass

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
    aos_result_num = int(os.environ.get("aos_results", ""))

    Kendra_index_id = os.environ.get("Kendra_index_id", "")
    Kendra_result_num = int(os.environ.get("Kendra_result_num", ""))
    # Opensearch_result_num = int(os.environ.get("Opensearch_result_num", ""))
    prompt_template = ''

    ##如果指定了prompt 模板
    if template_id and template_id != 'default':
        prompt_template = get_template(template_id)
        prompt_template = prompt_template['template']['S']
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
    
    main_entry_start = time.time()  # 或者使用 time.time_ns() 获取纳秒级别的时间戳
    answer,use_stream = main_entry_new(session_id, question, embedding_endpoint, llm_endpoint, model_name, aos_endpoint, aos_index, aos_knn_field, aos_result_num,
                       Kendra_index_id, Kendra_result_num,use_qa,wsclient,msgid,max_tokens,temperature,prompt_template,image_path,multi_rounds)
    main_entry_elpase = time.time() - main_entry_start  # 或者使用 time.time_ns() 获取纳秒级别的时间戳
    logger.info(f'runing time of main_entry : {main_entry_elpase}s seconds')
    # 2. return rusult

    # 处理

    # Response:
    # "id": "设置一个uuid"
    # "created": "1681891998"
    # "model": "模型名称"
    # "choices": [{"text": "模型回答的内容"}]
    # "usage": {"prompt_tokens": 58, "completion_tokens": 15, "total_tokens": 73}}]

    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': [{"id": str(uuid.uuid4()),
                  "use_stream":use_stream,
                             "created": request_timestamp,
                             "useTime": time.time() - request_timestamp,
                             "model": "main_brain",
                             "choices":
                             [{"text": "{}[{}]".format(answer, model_name)}],
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