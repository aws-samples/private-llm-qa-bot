import json
import logging
import time
import os
import re
import io
from botocore import config
from botocore.exceptions import ClientError,EventStreamError
from datetime import datetime, timedelta
import boto3
import uuid
import base64
from enum import Enum
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel,Field

import openai
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult
from typing import Any, Dict, List, Union,Mapping, Optional, TypeVar, Union
from langchain.schema import LLMResult
from enum import Enum
from boto3 import client as boto3_client
from utils.management import management_api,get_template
from utils.utils import add_reference, render_answer_with_ref
from generator.llm_wrapper import get_langchain_llm_model, invoke_model, format_to_message
from retriever.hybrid_retriever import CustomDocRetriever

lambda_client= boto3.client('lambda')
dynamodb_client = boto3.resource('dynamodb')
region = boto3.Session().region_name

DOC_INDEX_TABLE= 'chatbot_doc_index'
logger = logging.getLogger()
logger.setLevel(logging.INFO)
lambda_client = boto3_client('lambda')
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
SESSION_EXPIRES_DAYS = 1

BM25_QD_THRESHOLD_HARD_REFUSE = float(os.environ.get('bm25_qd_threshold_hard',15.0))
BM25_QD_THRESHOLD_SOFT_REFUSE = float(os.environ.get('bm25_qd_threshold_soft',20.0))
KNN_QQ_THRESHOLD_HARD_REFUSE = float(os.environ.get('knn_qq_threshold_hard',0.6))
KNN_QQ_THRESHOLD_SOFT_REFUSE = float(os.environ.get('knn_qq_threshold_soft',0.8))
KNN_QD_THRESHOLD_HARD_REFUSE = float(os.environ.get('knn_qd_threshold_hard',0.6))
KNN_QD_THRESHOLD_SOFT_REFUSE = float(os.environ.get('knn_qd_threshold_soft',0.8))
RERANK_THRESHOLD = float(os.environ.get('rerank_threshold_soft',-2))
WEBSEARCH_THRESHOLD = float(os.environ.get('websearch_threshold_soft',1))
CROSS_MODEL_ENDPOINT = os.environ.get('cross_model_endpoint',None)
KNN_QUICK_PEFETCH_THRESHOLD = float(os.environ.get('knn_quick_prefetch_threshold',0.95))

INTENTION_LIST = os.environ.get('intention_list', "")

TOP_K = int(os.environ.get('TOP_K',4))
NEIGHBORS = int(os.environ.get('neighbors',0))
KNOWLEDGE_BASE_ID = os.environ.get('knowledge_base_id',None)

BEDROCK_LLM_MODELID_LIST = {'claude-instant':'anthropic.claude-instant-v1',
                            'claude-v2':'anthropic.claude-v2',
                            'claude-v3-sonnet': 'anthropic.claude-3-sonnet-20240229-v1:0',
                            'claude-v3-haiku' : 'anthropic.claude-3-haiku-20240307-v1:0',
                            'llama3-70b': 'meta.llama3-70b-instruct-v1:0',
                            'llama3-8b': 'meta.llama3-8b-instruct-v1:0'}

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

def detect_intention(query, example_index='chatbot-example-index',fewshot_cnt=5):
    msg = {"fewshot_cnt":fewshot_cnt, "query": query,"example_index":example_index}
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

def chat_agent(query, detection):

    msg = {
      "params": {
        "query": query,
        "detection": detection 
      }
    }
    response = lambda_client.invoke(FunctionName="Chat_Agent",
                                           InvocationType='RequestResponse',
                                           Payload=json.dumps(msg))
    payload_json = json.loads(response.get('Payload').read())
    body = payload_json['body']
    answer = body['answer']
    ref_doc = body['ref_doc']

    return answer,ref_doc



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

        
def get_session(session_id,user_id):

    table_name = chat_session_table
    # dynamodb = boto3.resource('dynamodb')

    # table name
    table = dynamodb_client.Table(table_name)
    operation_result = ""
    try:
        response = table.get_item(Key={'session-id': session_id,'user_id':user_id})
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
def update_session(session_id,user_id,msgid, question, answer, intention):

    table_name = chat_session_table
    # dynamodb = boto3.resource('dynamodb')

    # table name
    table = dynamodb_client.Table(table_name)
    operation_result = ""

    response = table.get_item(Key={'session-id': session_id,'user_id':user_id})

    if "Item" in response.keys():
        # print("****** " + response["Item"]["content"])
        chat_history = json.loads(response["Item"]["content"])
    else:
        # print("****** No result")
        chat_history = []

    timestamp_str = str(datetime.now())
    expire_at = int(time.time())+3600*24*SESSION_EXPIRES_DAYS #session expires in 1 days 
    
    chat_history = [item for item in chat_history if len(item) >=6 and item[5] > int(time.time())]
    
    chat_history.append([question, answer, intention,msgid,timestamp_str,expire_at])
    content = json.dumps(chat_history,ensure_ascii=False)

    # inserting values into table
    response = table.put_item(
        Item={
            'session-id': session_id,
            'user_id':user_id,
            'content': content,
            'last_updatetime':timestamp_str,
            'expire_at':expire_at
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
        if len(item.get('doc_meta','')) > 0:
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

def history_to_messages(inputs) -> str:
    res = []
    for human, ai in inputs:
        res.append({ "role" : "user", "content" : human})
        res.append({ "role" : "assistant", "content" : ai})
    return res

def create_qa_prompt_templete(prompt_template):
    if prompt_template == '':
        prompt_template_zh = \
"""Human: {system_role_prompt}{role_bot}Here is a query:
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

Please follow below requirements:{ask_user_prompt}
- Respond in the original language of the question.
- Please try you best to leverage the image and hyperlink provided in <information>, you need to keep them in Markdown format.
- Do not directly reference the content of <information> in your answer.
- Skip the preamble, go straight into the answer. The answers will strictly be based on relevant knowledge in <information>.
- if the information is empty or not relevant to user's query, then reponse don't know.

Assistant:"""
    else:
        prompt_template_zh = prompt_template + '{ask_user_prompt}'
    PROMPT = PromptTemplate(
        template=prompt_template_zh,
        partial_variables={'system_role_prompt':SYSTEM_ROLE_PROMPT},
        input_variables=["context",'question','chat_history', 'role_bot', 'ask_user_prompt']
    )
    return PROMPT

def create_chat_prompt_templete(prompt_template='', llm_model_name='claude'):
    PROMPT = None
    if llm_model_name.startswith('claude'):
        prompt_template_zh = """Human: {system_role_prompt}{role_bot}Here is the conversation history (between the user and you) prior to the question. It could be empty if there is no history:
<history> {chat_history} </history>
Here is the user’s question: <question> {question} </question>
How do you respond to the user’s question?
Think about your answer first before you respond.
Assistant:"""
        PROMPT = PromptTemplate(
            template=prompt_template_zh, 
            partial_variables={'system_role_prompt':SYSTEM_ROLE_PROMPT},
            input_variables=['question', 'chat_history','role_bot']
        )
    elif llm_model_name.startswith('llama3'):
        prompt_template_zh = """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            {system_role_prompt}{role_bot}<|eot_id|><|start_header_id|>user<|end_header_id|>
            Here is the conversation history (between the user and you) prior to the question. It could be empty if there is no history:
            <history> {chat_history} </history>
            Here is the user’s question: <question> {question} </question>
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
        PROMPT = PromptTemplate(
            template=prompt_template_zh,
            partial_variables={'system_role_prompt': SYSTEM_ROLE_PROMPT},
            input_variables=['question', 'chat_history', 'role_bot']
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

def get_reply_stratgy(recall_knowledge,refuse_strategy:str):
    if not recall_knowledge:##如果希望LLM利用自有知识回答，则改成LLM_ONLY,否则SAY_DONT_KNOW
        if refuse_strategy == 'SAY_DONT_KNOW':
            stratgy = ReplyStratgy.SAY_DONT_KNOW
        elif refuse_strategy == 'WITH_LLM':
            stratgy = ReplyStratgy.WITH_LLM
        else:
            stratgy = ReplyStratgy.LLM_ONLY 
        return stratgy

    ## 如果使用了rerank模型
    if CROSS_MODEL_ENDPOINT:
        rank_score = [item['rank_score'] for item in recall_knowledge]
        if max(rank_score) < RERANK_THRESHOLD:  ##如果所有的知识都不超过rank score阈值
            ##如果希望LLM利用自有知识回答，则改成LLM_ONLY，否则SAY_DONT_KNOW
            if refuse_strategy == 'SAY_DONT_KNOW':
                stratgy = ReplyStratgy.SAY_DONT_KNOW
            elif refuse_strategy == 'WITH_LLM':
                stratgy = ReplyStratgy.WITH_LLM
            else:
                stratgy = ReplyStratgy.LLM_ONLY 
            return stratgy
        else:
            return ReplyStratgy.WITH_LLM
    else:
        ##使用rerank之后，不需要这些策略
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
            
            
def main_entry_new(user_id:str,wsconnection_id:str,session_id:str, query_input:str, embedding_model_endpoint:str, llm_model_endpoint:str, llm_model_name:str, aos_endpoint:str, aos_index:str, aos_knn_field:str, aos_result_num:int, kendra_index_id:str, 
                   kendra_result_num:int,use_qa:bool,wsclient=None,msgid:str='',max_tokens:int = 2048,temperature:float = 0.1,template:str = '',images_base64:List[str] = None,multi_rounds:bool = False, hide_ref:bool = False,use_stream:bool=False,example_index:str='chatbot-example-index',use_search:bool=True,refuse_strategy:str = 'LLM_ONLY',refuse_answer:str = ''):
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
            "LLM_input": '',
            "use_search":use_search
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

    params = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p":0.95
    }

    if llm_model_name.startswith('claude') or llm_model_name.startswith('llama'):
        model_id = BEDROCK_LLM_MODELID_LIST.get(llm_model_name, BEDROCK_LLM_MODELID_LIST['claude-v3-sonnet'])
    else:
        model_id = llm_model_endpoint
    logger.info("current model_id : {}".format(model_id))
    llm = get_langchain_llm_model(model_id, params, region, llm_stream=use_stream, llm_callbacks=[])
    
    # 1. get_session
    start1 = time.time()
    session_history = get_session(session_id=session_id,user_id=user_id)

    chat_coversions = [ (item[0],item[1]) for item in session_history]

    elpase_time = time.time() - start1
    logger.info(f'running time of get_session : {elpase_time}s seconds')
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

        chat_history_msgs=[]
        TRACE_LOGGER.trace(f'**Rewrite: {origin_query} => {query_input}, elpase_time:{elpase_time_rewrite:.3f}**')
        logger.info(f'Rewrite: {origin_query} => {query_input}')
        #add history parameter

        chat_history= get_chat_history(chat_coversions[-2:])
        chat_history_msgs= history_to_messages(chat_coversions[-2:])
    else:
        chat_history = ''
        chat_history_msgs= []


    
    doc_retriever = CustomDocRetriever.from_endpoints(embedding_model_endpoint=embedding_model_endpoint,
                                    aos_endpoint= aos_endpoint,
                                    aos_index=aos_index)
    
    TRACE_LOGGER.trace(f'**Using LLM model : {llm_model_name}**')
    cache_answer = None
    if use_qa:
        before_prefetch = time.time()
        TRACE_LOGGER.trace(f'**Prefetching cache...**')
        cache_repsonses = doc_retriever.knn_quick_prefetch(query_input=query_input, prefetch_threshold=KNN_QUICK_PEFETCH_THRESHOLD)
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
        detection = detect_intention(query_input,example_index, fewshot_cnt=5)
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
    elif intention in ['comfort', 'transfer']:
        reply_stratgy = ReplyStratgy.OTHER
        TRACE_LOGGER.trace('**Answer:**')
        answer = ''
        if intention == 'comfort':
            answer = "不好意思没能帮到您，是否帮你转人工客服？"
        elif intention == 'transfer':
            answer = '立即为您转人工客服，请稍后'
        
        if use_stream:
            TRACE_LOGGER.postMessage(answer)

        recall_knowledge,opensearch_knn_respose,opensearch_query_response = [],[],[]
    elif intention in ['chat', 'assist']:##如果不使用QA
        TRACE_LOGGER.trace(f'**Using Non-RAG {intention}...**')
        TRACE_LOGGER.trace('**Answer:**')
        reply_stratgy = ReplyStratgy.LLM_ONLY
        prompt_template = None
        answer = ''

        # for message api
        sys_msg = {"role": "system", "content": SYSTEM_ROLE_PROMPT } if SYSTEM_ROLE_PROMPT else None
        msg_list = [sys_msg, *chat_history_msgs] if sys_msg else [*chat_history_msgs]
        msg = format_to_message(query=origin_query, image_base64_list=images_base64)
        msg_list.append(msg)

        # for prompt api
        prompt_template = create_chat_prompt_templete(llm_model_name=llm_model_name)
        prompt = prompt_template.format(question=origin_query,role_bot=B_Role,chat_history=chat_history)

        ai_reply = invoke_model(llm=llm, prompt=prompt, messages=msg_list, callbacks=[stream_callback])

        final_prompt = json.dumps(msg_list,ensure_ascii=False)
        answer = ai_reply.content
        
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
            recall_knowledge = doc_retriever.get_relevant_documents_from_bedrock(
                    knowledge_base_id=KNOWLEDGE_BASE_ID, 
                    query_input=query_input, 
                    top_k=TOP_K, 
                    rerank_endpoint=CROSS_MODEL_ENDPOINT, 
                    rerank_threshold=RERANK_THRESHOLD
                )
        else:
            TRACE_LOGGER.trace('**Retrieving knowledge from OpenSearch...**')
            recall_knowledge, opensearch_knn_respose, opensearch_query_response = doc_retriever.get_relevant_documents_custom(
                query_input=query_input, 
                channel_return_cnt=CHANNEL_RET_CNT, 
                top_k=TOP_K, 
                knn_threshold=KNN_QQ_THRESHOLD_HARD_REFUSE, 
                bm25_threshold=BM25_QD_THRESHOLD_HARD_REFUSE, 
                web_search_threshold=WEBSEARCH_THRESHOLD, 
                use_search=use_search,
                rerank_endpoint=CROSS_MODEL_ENDPOINT,
                rerank_threshold=RERANK_THRESHOLD
            ) 

        elpase_time = time.time() - start
        logger.info(f'running time of opensearch_query : {elpase_time:.3f}s seconds')
        
        reply_stratgy = get_reply_stratgy(recall_knowledge,refuse_strategy)
        if reply_stratgy == ReplyStratgy.LLM_ONLY: 
            recall_knowledge = []
            
        TRACE_LOGGER.trace(f'**Running time of retrieving knowledge : {elpase_time:.3f}s**')
        TRACE_LOGGER.trace(f'**Retrieved {len(recall_knowledge)} knowledge:**')
        TRACE_LOGGER.add_ref(f'\n\n**Refer to {len(recall_knowledge)} knowledge:**')

        ##添加召回文档到refdoc和tracelog, 按score倒序展示
        for sn,item in enumerate(recall_knowledge[::-1]):
            TRACE_LOGGER.trace(f"**[{sn+1}] [{item['doc_title']}] [{item['doc_classify']}] [{item['score']:.3f}] [{item['rank_score']:.3f}] author:[ {item['doc_author']} ]**")
            TRACE_LOGGER.trace(f"{item['doc']}")
            TRACE_LOGGER.add_ref(f"**[{sn+1}] [{item['doc_title']}] [{item['doc_classify']}] [{item['score']:.3f}] [{item['rank_score']:.3f}] author:[ {item['doc_author']} ]**")
            #doc 太长之后进行截断
            TRACE_LOGGER.add_ref(f"{item['doc'][:500]}{'...' if len(item['doc'])>500 else ''}") 
        TRACE_LOGGER.trace('**Answer:**')

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
            answer = refuse_answer
            hide_ref= True ## 隐藏ref doc
            if use_stream:
                TRACE_LOGGER.postMessage(answer)
                
        elif reply_stratgy == ReplyStratgy.LLM_ONLY: ##走LLM默认知识
            # prompt_template = create_chat_prompt_templete()
            # hide_ref= True ## 隐藏ref doc
            # llmchain = LLMChain(llm=llm,verbose=verbose,prompt =prompt_template )
            # ##最终的answer
            # answer = llmchain.run({'question':query_input,'chat_history':chat_history,'role_bot':B_Role})
            # ##最终的prompt日志
            
            # for message api
            sys_msg = {"role": "system", "content": SYSTEM_ROLE_PROMPT } if SYSTEM_ROLE_PROMPT else None
            msg_list = [sys_msg, *chat_history_msgs] if sys_msg else [*chat_history_msgs]
            msg = format_to_message(query=origin_query, image_base64_list=images_base64)
            msg_list.append(msg)

            # for prompt api
            prompt_template = create_chat_prompt_templete(llm_model_name=llm_model_name)
            prompt = prompt_template.format(question=origin_query,role_bot=B_Role,chat_history=chat_history)

            ai_reply = invoke_model(llm=llm, prompt=prompt, messages=msg_list, callbacks=[stream_callback])

            final_prompt = json.dumps(msg_list,ensure_ascii=False)
            answer = ai_reply.content
            
        else:      
            prompt_template = create_qa_prompt_templete(template)
            # llmchain = LLMChain(llm=llm,verbose=verbose,prompt =prompt_template )

            # context = "\n".join([doc['doc'] for doc in recall_knowledge])
            context, multi_choice_field = format_knowledges(recall_knowledge)
            ask_user_prompts = [ f"- If you are not sure about which {field} user ask for, please ask user to clarify it before giving any answer, don't say anything else." for field in multi_choice_field ]
            ask_user_prompts_str = "\n".join(ask_user_prompts)
            if len(ask_user_prompts_str) > 0:
                ask_user_prompts_str = f"\n{ask_user_prompts_str}\n"

            try:
                chat_history = '' ##QA 场景下先不使用history
                prompt = prompt_template.format(question=query_input,role_bot=B_Role,context=context,chat_history=chat_history,ask_user_prompt=ask_user_prompts_str)
                
                sys_msg = {"role": "system", "content": SYSTEM_ROLE_PROMPT } if SYSTEM_ROLE_PROMPT else None
                msg_list = [sys_msg, *chat_history_msgs] if sys_msg else [*chat_history_msgs]
                msg = format_to_message(query=prompt, image_base64_list=images_base64)
                msg_list.append(msg)

                ai_reply = invoke_model(llm=llm, prompt=prompt, messages=msg_list, callbacks=[stream_callback])
                final_prompt = json.dumps(msg_list,ensure_ascii=False)
                answer = ai_reply.content
            except Exception as e:
                answer = str(e)
    else:
        #call agent for other intentions
        TRACE_LOGGER.trace('**Using Agent...**')
        reply_stratgy = ReplyStratgy.AGENT

        answer,ref_doc = chat_agent(query_input, detection)
        recall_knowledge,opensearch_knn_respose,opensearch_query_response = [ref_doc],[],[]
        TRACE_LOGGER.add_ref(f'\n\n**Refer to {len(recall_knowledge)} knowledge:**')
        TRACE_LOGGER.add_ref(f"**[1]** {ref_doc}")
        TRACE_LOGGER.trace(f'**Function call result:**\n\n{ref_doc}')
        TRACE_LOGGER.trace('**Answer:**')
        if use_stream:
            TRACE_LOGGER.postMessage(answer)

    answer = enforce_stop_tokens(answer, STOP)
    pattern = r'^根据[^，,]*[,|，]'
    answer = re.sub(pattern, "", answer.strip())
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
        "reply_stratgy" : reply_stratgy.name,
        "use_search":use_search,
        "aos_index":aos_index,
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
        update_session(session_id=session_id,user_id=user_id, question=origin_query, answer=answer, intention=intention, msgid=msgid)
    elpase_time = time.time() - start
    elpase_time1 = time.time() - start1
    logger.info(f'running time of update_session : {elpase_time}s seconds')
    logger.info(f'running time of all  : {elpase_time1}s seconds')
    return answer,ref_text,use_stream,query_input,opensearch_query_response,opensearch_knn_respose,recall_knowledge

def get_s3_image_base64(bucket_name, key):
    # Create an S3 client
    s3 = boto3.client('s3')
    # Get the object from S3
    try:
        response = s3.get_object(Bucket=bucket_name, Key=key)
        image_data = response['Body'].read()
        # Encode the image data as base64
        base64_image = base64.b64encode(image_data).decode('utf-8')
        return base64_image
    except Exception as e:
        print(f"Error getting object from S3: {e}")
        return None

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
    global CHANNEL_RET_CNT
    CHANNEL_RET_CNT = event.get('channel_cnt', 10)
    logger.info(f'channel_cnt:{CHANNEL_RET_CNT}')

    ###其他管理操作 start
    ###其他管理操作 start
    if resource:
        ret_json = management_api(method,resource,event)
        return ret_json

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
    feature_config = event.get('feature_config','')
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
    aos_index = os.environ.get("aos_index", "")
    company = event.get("company",'default')
    example_index = "chatbot-example-index"
    aos_index = f'chatbot-index-{company}'
    example_index = f'chatbot-example-index-{company}'
    refuse_strategy = event.get('refuse_strategy','')
    refuse_answer = event.get('refuse_answer','对不起，我不太清楚这个问题，请问问人工吧')
        
    
    imgurls = event.get('imgurl')
    image_path = ''
    images_base64 = []
    if imgurls:
        logger.info(f"imgurls:{imgurls}")
        for imgurl in imgurls:
            bucket,imgobj = imgurl.split('/',1)
            # image_path = generate_s3_image_url(bucket,imgobj)
            image_base64 = get_s3_image_base64(bucket,imgobj)
            images_base64.append(image_base64)

    ## 用于trulength接口，只返回recall 知识
    if retrieve_only:
        doc_retriever = CustomDocRetriever.from_endpoints(embedding_model_endpoint=embedding_endpoint,
                                    aos_endpoint= os.environ.get("aos_endpoint", ""),
                                    aos_index=aos_index)
        recall_knowledge,opensearch_knn_respose,opensearch_query_response = doc_retriever.get_relevant_documents_custom(
                query_input=question, 
                channel_return_cnt=CHANNEL_RET_CNT, 
                top_k=TOP_K, 
                knn_threshold=KNN_QQ_THRESHOLD_HARD_REFUSE, 
                bm25_threshold=BM25_QD_THRESHOLD_HARD_REFUSE, 
                web_search_threshold=WEBSEARCH_THRESHOLD, 
                use_search=use_search,
                rerank_endpoint=CROSS_MODEL_ENDPOINT,
                rerank_threshold=RERANK_THRESHOLD
            ) 
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

    # 接收触发AWS Lambda函数的事件
    logger.info('The main brain has been activated, aws🚀!')

    # 1. 获取环境变量

    # embedding_endpoint = os.environ.get("embedding_endpoint", "")
    aos_endpoint = os.environ.get("aos_endpoint", "")
    aos_knn_field = os.environ.get("aos_knn_field", "")
    aos_result_num = int(os.environ.get("aos_results", 4))

    Kendra_index_id = os.environ.get("Kendra_index_id", "")
    Kendra_result_num = int(os.environ.get("Kendra_result_num", 0))
    # Opensearch_result_num = int(os.environ.get("Opensearch_result_num", ""))
    prompt_template = ''

    ##如果指定了prompt 模板
    if template_id and template_id != 'default':
        prompt_template = get_template(template_id,company)
        prompt_template = '' if prompt_template is None else prompt_template['template']['S']
    
    use_search = False if feature_config == 'search_disabled' else True
    logger.info(f'use_search : {use_search}')
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
    logger.info(f'refuse_strategy: {refuse_strategy}')
    logger.info(f'refuse_answer: {refuse_answer}')
    
    ##if aos and bedrock kb are null then set use_qa = false
    if not aos_endpoint and not KNOWLEDGE_BASE_ID:
        use_qa = False
        logger.info(f'force set use_qa: {use_qa}')
        
    global TRACE_LOGGER
    TRACE_LOGGER = TraceLogger(wsclient=wsclient,msgid=msgid,connectionId=wsconnection_id,stream=use_stream,use_trace=use_trace,hide_ref=hide_ref)
    main_entry_start = time.time()  # 或者使用 time.time_ns() 获取纳秒级别的时间戳
    answer,ref_text,use_stream,query_input,opensearch_query_response,opensearch_knn_respose,recall_knowledge = main_entry_new(user_id,wsconnection_id,session_id, question, embedding_endpoint, llm_endpoint, model_name, aos_endpoint, aos_index, aos_knn_field, aos_result_num,
                       Kendra_index_id, Kendra_result_num,use_qa,wsclient,msgid,max_tokens,temperature,prompt_template,images_base64,multi_rounds,hide_ref,use_stream,example_index,use_search,refuse_strategy,refuse_answer)
    main_entry_elpase = time.time() - main_entry_start  # 或者使用 time.time_ns() 获取纳秒级别的时间戳
    logger.info(f'running time of main_entry : {main_entry_elpase}s seconds')
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
        extra_info = {"query_input": query_input, "opensearch_query_response" : opensearch_query_response, "opensearch_knn_respose": opensearch_knn_respose,"recall_knowledge":recall_knowledge }
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': [{"id": str(uuid.uuid4()),
                  "use_stream":use_stream,
                  "query":query_input,
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