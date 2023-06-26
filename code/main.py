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
from typing import List
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from typing import Dict, List
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain import PromptTemplate, SagemakerEndpoint
from langchain.chains import LLMChain,ConversationalRetrievalChain,ConversationChain
from langchain.schema import BaseRetriever
from langchain.schema import Document
from langchain.llms.bedrock import Bedrock
from pydantic import BaseModel
import openai
from langchain.callbacks.base import BaseCallbackHandler
import sys
from typing import Any, Dict, List, Union
from langchain.schema import LLMResult




class CustomStreamingOutCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming. Only works with LLMs that support streaming."""
    def __init__(self,wsclient:str,msgid:str,connectionId:str , **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.wsclient = wsclient
        self.connectionId = connectionId
        self.msgid = msgid

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
        data = json.dumps({ 'msgid':self.msgid, 'role': "AI", 'text': {'content':'[DONE]'} })
        self.postMessage(data)

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        data = json.dumps({ 'msgid':self.msgid, 'role': "AI", 'text': {'content':str(error[0])+'[DONE]'} })
        self.postMessage(data)

credentials = boto3.Session().get_credentials()
region = boto3.Session().region_name
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, 'es', session_token=credentials.token)


logger = logging.getLogger()
logger.setLevel(logging.INFO)
sm_client = boto3.client("sagemaker-runtime")
# llm_endpoint = 'bloomz-7b1-mt-2023-04-19-09-41-24-189-endpoint'
chat_session_table = os.environ.get('chat_session_table')
QA_SEP = "=>"
A_Role="用户"
B_Role="AWSBot"
SYSTEM_ROLE_PROMPT = '你是云服务AWS的智能客服机器人AWSBot'
Fewshot_prefix_Q="问题"
Fewshot_prefix_A="回答"
RESET = '/rs'


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
    parameters = {
        "max_length": 2048,
        "temperature": 0.01,
        "num_beams": 1, # >1可能会报错，"probability tensor contains either `inf`, `nan` or element < 0"； 即使remove_invalid_values=True也不能解决
        "do_sample": False,
        "top_p": 0.7,
    }
    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({'inputs': prompt,'history':[],**model_kwargs})
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
        recall_knowledge,_,_ = self.get_relevant_documents_custom(query_input)
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
            def get_topk_items(opensearch_knn_respose, opensearch_query_response, doc_type, topk=1):
                opensearch_knn_items = [ (item['doc'], item['score']) for item in opensearch_knn_respose ]
                opensearch_bm25_items = [ (item['doc'], item['score']) for item in opensearch_query_response ]

                opensearch_knn_items.sort(key=lambda x: x[1])
                opensearch_bm25_items.sort(key=lambda x: x[1])

                opensearch_knn_nodup = [(k,v) for k,v in dict(opensearch_knn_items).items()]
                opensearch_bm25_nodup = [(k,v) for k,v in dict(opensearch_bm25_items).items()]

                opensearch_knn_nodup.sort(key=lambda x: x[1])
                opensearch_bm25_nodup.sort(key=lambda x: x[1])

                kg_combine_result = [ { "doc": item[0], "score": item[1], "doc_type": doc_type } for item in opensearch_knn_nodup[-1*topk:]]
                knn_kept_doc = [ item[0] for item in opensearch_knn_nodup[-1*topk:] ]

                count = 0
                for item in opensearch_bm25_nodup[::-1]:
                    if item[0] not in knn_kept_doc:
                        kg_combine_result.append({ "doc": item[0], "score": item[1], "doc_type": doc_type })
                        count += 1
                    if count == topk:
                        break

                return kg_combine_result


            knn_threshold = 0.2
            inverted_theshold = 5.0
            filter_knn_result = [ item for item in opensearch_knn_respose if item['doc_type'] in ['Paragraph','Sentence'] and item['score'] > knn_threshold ]
            filter_inverted_result = [ item for item in opensearch_query_response if item['doc_type'] in ['Paragraph','Sentence'] and item['score'] > inverted_theshold ]
            
            paragraph_content = get_topk_items(filter_knn_result, filter_inverted_result, "Paragraph", 2)

            knn_threshold = 0.2
            inverted_theshold = 5.0
            filter_knn_result = [ item for item in opensearch_knn_respose if item['doc_type'] == 'Question' and item['score'] > knn_threshold ]
            filter_inverted_result = [ item for item in opensearch_knn_respose if item['doc_type'] == 'Question' and item['score'] > inverted_theshold ]

            qa_content = get_topk_items(filter_knn_result, filter_inverted_result, "Question", 2)

            ret_content = paragraph_content + qa_content
            return ret_content
        
        def combine_union_recalls(opensearch_knn_respose, opensearch_query_response):
            '''
            filter knn_result if the result don't appear in filter_inverted_result
            '''
            knn_threshold = 0.2
            inverted_theshold = 5.0
            filter_knn_result = { item["id"] :( item["doc"],item["score"]) for item in opensearch_knn_respose if item["score"]> knn_threshold }
            filter_inverted_result = { item["id"] :( item["doc"],item["score"]) for item in opensearch_query_response if item["score"]> inverted_theshold }

            combine_result = []
            
            for key, items in (filter_knn_result|filter_inverted_result).items():
                combine_result.append({ "doc" : items[0], "score" : items[1] })
            return combine_result
        
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


# AOS
def get_vector_by_sm_endpoint(questions, sm_client, endpoint_name):
    parameters = {
        # "early_stopping": True,
        # "length_penalty": 2.0,
        "max_new_tokens": 50,
        "temperature": 0,
        "min_length": 10,
        "no_repeat_ngram_size": 2,
    }

    response_model = sm_client.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=json.dumps(
            {
                "inputs": questions,
                "parameters": parameters
            }
        ),
        ContentType="application/json",
    )
    json_str = response_model['Body'].read().decode('utf8')
    json_obj = json.loads(json_str)
    embeddings = json_obj['sentence_embeddings']
    return embeddings

def search_using_aos_knn(client, q_embedding, index, size=10):
    # awsauth = (username, passwd)
    # print(type(q_embedding))
    # logger.info(f"q_embedding:")
    # logger.info(q_embedding)
    headers = {"Content-Type": "application/json"}
    # query = {
    #     "size": size,
    #     "query": {
    #         "bool": {
    #             "must":[ {"term": { "doc_type": "P" }} ],
    #             "should": [ { "knn": { "embedding": { "vector": q_embedding, "k": size }}} ]
    #         }
    #     },
    #     "sort": [
    #         {
    #             "_score": {
    #                 "order": "asc"
    #             }
    #         }
    #     ]
    # }

    # reference: https://opensearch.org/docs/latest/search-plugins/knn/filter-search-knn/#boolean-filter-with-ann-search
    # query =  {
    #     "bool": {
    #         "filter": {
    #             "bool": {
    #                 "must": [{ "term": {"doc_type": "P" }}]
    #             }
    #         },
    #         "must": [
    #             {
    #                 "knn": {"embedding": { "vector": q_embedding, "k": size }}
    #             } 
    #         ]
    #     }
    # }


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
    opensearch_knn_respose = [{'id':item['_id'],'doc':"{}{}{}".format(item['_source']['doc'], QA_SEP, item['_source']['content']),"doc_type":item["_source"]["doc_type"],"score":item["_score"]}  for item in query_response["hits"]["hits"]]
    return opensearch_knn_respose
    # try:
    #     r = requests.post("https://"+hostname + "/" + index +
    #                     '/_search', headers=headers, json=query)
    #     results = json.loads(r.text)["hits"]["hits"]
    #     for item in results:
    #         opensearch_knn_respose.append( {'id':item['_id'],'doc':"{}{}{}".format(item['_source']['doc'], QA_SEP, item['_source']['answer']),"doc_type":item["_source"]["doc_type"],"score":item["_score"]} )
    #     return opensearch_knn_respose
    # except Exception as e:
    #     print(f'knn query exception:{str(e)}')
    #     return []
    


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
                    "doc": query_term
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
                                            "doc": query_term
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
        result_arr = [ {'id':item['_id'],'doc': item['_source']['content'], 'doc_type': item['_source']['doc_type'], 'score': item['_score']} for item in query_response["hits"]["hits"]]
    else:
        result_arr = [ {'id':item['_id'],'doc':"{}{}{}".format(item['_source']['doc'], QA_SEP, item['_source']['content']), 'doc_type': item['_source']['doc_type'], 'score': item['_score']} for item in query_response["hits"]["hits"]]

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

    response = table.get_item(Key={'session-id': session_id})

    if "Item" in response.keys():
        # print("****** " + response["Item"]["content"])
        operation_result = json.loads(response["Item"]["content"])
    else:
        # print("****** No result")
        operation_result = ""

    return operation_result


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

def Generate(smr_client, llm_endpoint, prompt, llm_name, stop=None, history=[]):
    answer = None
    if llm_name == "chatglm":
        logger.info("call chatglm...")
        parameters = {
        "max_length": 2048,
        "temperature": 0.01,
        "num_beams": 1, # >1可能会报错，"probability tensor contains either `inf`, `nan` or element < 0"； 即使remove_invalid_values=True也不能解决
        "do_sample": False,
        "top_p": 0.7,
        "logits_processor" : None,
        # "remove_invalid_values" : True
        }
        response_model = smr_client.invoke_endpoint(
            EndpointName=llm_endpoint,
            Body=json.dumps(
            {
                "inputs": prompt,
                "parameters": parameters,
                "history" : history
            }
            ),
            ContentType="application/json",
        )

        json_ret = json.loads(response_model['Body'].read().decode('utf8'))

        answer = json_ret['outputs']
    elif llm_name == "bloomz":
        logger.info("call bloomz...")
        parameters = {
            # "early_stopping": True,
            "length_penalty": 1.0,
            "max_new_tokens": 200,
            "temperature": 0,
            "min_length": 20,
            "no_repeat_ngram_size": 200,
            # "eos_token_id": ['\n']
        }

        response_model = smr_client.invoke_endpoint(
            EndpointName=llm_endpoint,
            Body=json.dumps(
                {
                    "inputs": prompt,
                    "parameters": parameters
                }
            ),
            ContentType="application/json",
        )
        
        json_ret = json.loads(response_model['Body'].read().decode('utf8'))
        answer = json_ret['outputs'][len(prompt):]

    return enforce_stop_tokens(answer, stop)


class QueryType(Enum):
    KeywordQuery   = "KeywordQuery"       #用户仅仅输入了一些关键词（2 token)
    KnowledgeQuery = "KnowledgeQuery"     #用户输入的需要参考知识库有关来回答
    Conversation   = "Conversation"       #用户输入的是跟知识库无关的问题


# def intention_classify(post_text, prompt_template, few_shot_example):
#     prompt = prompt_template.format(
#         fewshot=few_shot_example, question=post_text)
#     result = Generate(sm_client, llm_endpoint, prompt)
#     return result


def qa_knowledge_fewshot_build(recalls):
    ret_context = []
    for recall in recalls:
        if recall['doc_type'] == 'Question':
            q, a = recall['doc'].split(QA_SEP)
            qa_example = "{}: {}\n{}: {}".format(Fewshot_prefix_Q, q, Fewshot_prefix_A, a)
            ret_context.append(qa_example)
        elif recall['doc_type'] == 'Paragraph':
            recall_doc, p = recall['doc'].split(QA_SEP)
            p_example = "{}".format(p)
            ret_context.append(p_example)

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

def create_qa_prompt_templete(lang='zh'):
    if lang == 'zh':
        prompt_template_zh = """{system_role_prompt}，请严格根据反括号中的资料提取相关信息，回答{role_user}的各种问题\n```\n{chat_history}{context}\n```\n\n{role_user}: {question}\n{role_bot}: """

        PROMPT = PromptTemplate(
            template=prompt_template_zh,
            partial_variables={'system_role_prompt':SYSTEM_ROLE_PROMPT},
            input_variables=["context",'question','chat_history','role_bot','role_user']
        )
    return PROMPT

def create_chat_prompt_templete(lang='zh'):
    if lang == 'zh':
        prompt_template_zh = """{system_role_prompt}，能够回答{role_user}的各种问题以及陪{role_user}聊天,如:{chat_history}\n\n{role_user}: {question}\n{role_bot}:"""

        PROMPT = PromptTemplate(
            template=prompt_template_zh, 
            partial_variables={'system_role_prompt':SYSTEM_ROLE_PROMPT},
            input_variables=['question','chat_history','role_bot','role_user']
        )
    return PROMPT

def get_bedrock_secret(secret_name='chatbot_bedrock', region_name = "us-west-2"):
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
    return secret['BEDROCK_SECRET_KEY']

def main_entry_new(session_id:str, query_input:str, embedding_model_endpoint:str, llm_model_endpoint:str, llm_model_name:str, aos_endpoint:str, aos_index:str, aos_knn_field:str,
                    aos_result_num:int, kendra_index_id:str, kendra_result_num:int,use_qa:bool,wsclient=None,msgid=''):
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
        return answer
    
    print("llm_model_name : {}".format(llm_model_name))
    llm = None
    if llm_model_name == 'claude':
        ACCESS_KEY='AKIA2EEYKWL2VR6DKVEI'
        SECRET_KEY=get_bedrock_secret()

        boto3_bedrock = boto3.client(
            service_name="bedrock",
            region_name="us-east-1",
            endpoint_url="https://bedrock.us-east-1.amazonaws.com",
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY
        )
        llm = Bedrock(model_id="anthropic.claude-v1", client=boto3_bedrock)
        # print("llm is anthropic.claude-v1")
    elif llm_model_name.startswith('gpt-3.5-turbo'):
        global openai_api_key
        llm=ChatOpenAI(model = llm_model_name,
                       openai_api_key = openai_api_key,
                       streaming = True,
                       callbacks=[CustomStreamingOutCallbackHandler(wsclient,msgid, session_id,)],
                       temperature = 0.01)
        use_stream = True
    else:
        llmcontent_handler = llmContentHandler()
        llm=SagemakerEndpoint(
                endpoint_name=llm_model_endpoint, 
                region_name=region, 
                model_kwargs={'parameters':llmcontent_handler.parameters},
                content_handler=llmcontent_handler
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
    free_chat_coversions = []
    verbose = False
    if not use_qa:##如果不使用QA
        query_type = QueryType.Conversation
        free_chat_coversions = [ (item[0],item[1]) for item in session_history if item[2] == QueryType.Conversation]
        chat_history= get_chat_history(free_chat_coversions[-2:])
        prompt_template = create_chat_prompt_templete(lang='zh')
        llmchain = LLMChain(llm=llm,verbose=verbose,prompt =prompt_template )
        ##最终的answer
        answer = llmchain.run({'question':query_input,'chat_history':chat_history,'role_bot':B_Role,'role_user':A_Role})
        ##最终的prompt日志
        final_prompt = prompt_template.format(question=query_input,role_bot=B_Role,role_user=A_Role,chat_history=chat_history)
        recall_knowledge,opensearch_knn_respose,opensearch_query_response = [],[],[]
    else: ##如果使用QA
        # 2. aos retriever
        doc_retriever = CustomDocRetriever.from_endpoints(embedding_model_endpoint=embedding_model_endpoint,
                                    aos_endpoint= aos_endpoint,
                                    aos_index=aos_index)
        # 3. check is it keyword search
        exactly_match_result = aos_search(aos_endpoint, aos_index, "doc", query_input, exactly_match=True)

        start = time.time()
        ## 加上一轮的问题拼接来召回内容
        # query_with_history= get_question_history(chat_coversions[-2:])+query_input
        recall_knowledge,opensearch_knn_respose,opensearch_query_response = doc_retriever.get_relevant_documents_custom(query_input) 
        elpase_time = time.time() - start
        logger.info(f'runing time of opensearch_query : {elpase_time}s seconds')

        if exactly_match_result and recall_knowledge: 
            query_type = QueryType.KeywordQuery
            answer = exactly_match_result[0]["doc"]
            final_prompt = ''
        elif recall_knowledge:      
            # chat_history= get_chat_history(chat_coversions[-2:]) ##chatglm模型质量不高，暂时屏蔽历史对话
            chat_history = ''
            query_type = QueryType.KnowledgeQuery
            prompt_template = create_qa_prompt_templete(lang='zh')
            llmchain = LLMChain(llm=llm,verbose=verbose,prompt =prompt_template )
            # context = "\n".join([doc['doc'] for doc in recall_knowledge])
            context = qa_knowledge_fewshot_build(recall_knowledge)
            ##最终的answer
            answer = llmchain.run({'question':query_input,'context':context,'chat_history':chat_history,'role_bot':B_Role,'role_user':A_Role})
            ##最终的prompt日志
            final_prompt = prompt_template.format(question=query_input,role_bot=B_Role,role_user=A_Role,context=context,chat_history=chat_history)
            # print(final_prompt)
            # print(answer)
        else:
            query_type = QueryType.Conversation
            free_chat_coversions = [ (item[0],item[1]) for item in session_history if item[2] == QueryType.Conversation ]
            # free_chat_coversions = [ (item[0],item[1]) for item in session_history ]
            chat_history= get_chat_history(free_chat_coversions[-2:])
            prompt_template = create_chat_prompt_templete(lang='zh')
            llmchain = LLMChain(llm=llm,verbose=verbose,prompt =prompt_template )
            ##最终的answer
            answer = llmchain.run({'question':query_input,'chat_history':chat_history,'role_bot':B_Role,'role_user':A_Role})
            ##最终的prompt日志
            final_prompt = prompt_template.format(question=query_input,role_bot=B_Role,role_user=A_Role,chat_history=chat_history)

    STOP=[f"\n{A_Role}", f"\n{B_Role}"]
    answer = enforce_stop_tokens(answer, STOP)

    json_obj = {
        "query": query_input,
        "opensearch_doc":  opensearch_query_response,
        "opensearch_knn_doc":  opensearch_knn_respose,
        "kendra_doc": [],
        "knowledges" : recall_knowledge,
        "detect_query_type": str(query_type),
        "LLM_input": final_prompt
    }

    json_obj['session_id'] = session_id
    json_obj['chatbot_answer'] = answer
    json_obj['conversations'] = free_chat_coversions
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
    # dynamodb = boto3.client('dynamodb')
    # try:
    #     dynamodb.delete_item(
    #         TableName=DOC_INDEX_TABLE,
    #         Key={
    #             'filename': {'S': obj_key},
    #             'embedding_model': {'S': embedding_model}
    #         }
    #     )
    # except Exception as e:
    #     logger.info(str(e))

    ##删除aos里的索引
    should_continue = True
    while should_continue:
        should_continue = delete_aos_index(obj_key,index_name)
    


@handle_error
def lambda_handler(event, context):
    # "model": 模型的名称
    # "chat_name": 对话标识，后端用来存储查找实现多轮对话 session
    # "prompt": 用户输入的问题
    # "max_tokens": 2048
    # "temperature": 0.9
    logger.info(f"event:{event}")
    delete_obj_key = event.get('delete_obj_key')
    ##如果是删除doc index的操作
    if delete_obj_key:
        delete_doc_index(delete_obj_key,os.environ.get("aos_index", ""))
        return {'statusCode': 200}

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
    embedding_endpoint = event['embedding_model'] 
    use_qa = event.get('use_qa',True)
    msgid = event.get('msgid')

    ##获取前端给的系统设定，如果没有，则使用lambda里的默认值
    global B_Role,SYSTEM_ROLE_PROMPT
    B_Role = event.get('system_role') if event.get('system_role') else B_Role
    SYSTEM_ROLE_PROMPT = event.get('system_role_prompt') if event.get('system_role_prompt') else SYSTEM_ROLE_PROMPT

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
    else:
        llm_endpoint = os.environ.get('llm_default_endpoint')
        pass

    # 获取当前时间戳
    request_timestamp = time.time()  # 或者使用 time.time_ns() 获取纳秒级别的时间戳
    logger.info(f'request_timestamp :{request_timestamp}')
    logger.info(f"event:{event}")
    logger.info(f"context:{context}")

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

    logger.info(f'model_name : {model_name}')
    logger.info(f'llm_endpoint : {llm_endpoint}')
    logger.info(f'embedding_endpoint : {embedding_endpoint}')
    logger.info(f'aos_endpoint : {aos_endpoint}')
    logger.info(f'aos_index : {aos_index}')
    logger.info(f'aos_knn_field : {aos_knn_field}')
    logger.info(f'aos_result_num : {aos_result_num}')
    logger.info(f'Kendra_index_id : {Kendra_index_id}')
    logger.info(f'Kendra_result_num : {Kendra_result_num}')
    
    main_entry_start = time.time()  # 或者使用 time.time_ns() 获取纳秒级别的时间戳
    answer,use_stream = main_entry_new(session_id, question, embedding_endpoint, llm_endpoint, model_name, aos_endpoint, aos_index, aos_knn_field, aos_result_num,
                       Kendra_index_id, Kendra_result_num,use_qa,wsclient,msgid)
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
                             [{"text": answer}],
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

