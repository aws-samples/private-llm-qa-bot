
import json
import logging
import time
import os
import re
from botocore import config
from botocore.exceptions import ClientError,EventStreamError
from datetime import datetime, timedelta
import boto3
import time
import hashlib
import uuid
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
import logger
from utils.web_search import web_search,add_webpage_content
from utils.management import management_api,get_template
from utils.utils import add_reference, render_answer_with_ref

lambda_client= boto3.client('lambda')
dynamodb_client = boto3.resource('dynamodb')
credentials = boto3.Session().get_credentials()
region = boto3.Session().region_name
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, 'es', session_token=credentials.token)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    logger.info(f"event:{event}")
    request_timestamp = time.time()  
    use_stream = event.get('use_stream',False)
    query_input = event['prompt']
    openai_api_key = event.get('OPENAI_API_KEY') 
    hide_ref = event.get('hide_ref',False)
    feature_config = event.get('feature_config','')
    retrieve_only = event.get('retrieve_only',False)
    session_id = event['chat_name']
    wsconnection_id = event.get('wsconnection_id',session_id)
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
    
    answer = 'placeholder'
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
                "extra_info" : '',
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
            ]
    }