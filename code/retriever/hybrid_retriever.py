import json
import boto3
import time
import logging
import hashlib
import math
from opensearchpy import OpenSearch, RequestsHttpConnection
from typing import Any, Dict, List, Union,Mapping, Optional, TypeVar, Union
from requests_aws4auth import AWS4Auth
from langchain.schema import BaseRetriever
from langchain.schema import Document
from .web_search import web_search, add_webpage_content

logger = logging.getLogger()
logger.setLevel(logging.INFO)

credentials = boto3.Session().get_credentials()
region = boto3.Session().region_name
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, 'es', session_token=credentials.token)
boto3_bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name= region
)
sm_client = boto3.client("sagemaker-runtime")
knowledgebase_client = boto3.client("bedrock-agent-runtime", region)

BEDROCK_EMBEDDING_MODELID_LIST = [ 
    "cohere.embed-multilingual-v3",
    "cohere.embed-english-v3",
    "amazon.titan-embed-text-v1"
]

def is_chinese(string):
    for char in string:
        if '\u4e00' <= char <= '\u9fff':
            return True

    return False

def get_embedding_bedrock(texts, model_id):
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
    opensearch_knn_respose = [{'idx':item['_source'].get('idx',1),'doc_category':item['_source']['doc_category'],'doc_classify':item['_source'].get('doc_classify'),'doc_title':item['_source']['doc_title'],'id':item['_id'],'doc':item['_source']['content'],"doc_type":item["_source"]["doc_type"],"score":item["_score"],'doc_author': item['_source']['doc_author'], 'doc_meta': item['_source'].get('doc_meta','')}  for item in query_response["hits"]["hits"]]
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
                "match": { "content" : query_term }
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
        result_arr = [ {'idx':item['_source'].get('idx',0),'doc_category':item['_source']['doc_category'],'doc_classify':item['_source'].get('doc_classify'),'doc_title':item['_source']['doc_title'],'id':item['_id'],'doc': item['_source']['content'], 'doc_type': item['_source']['doc_type'], 'score': item['_score'],'doc_author': item['_source']['doc_author'], 'doc_meta': item['_source'].get('doc_meta','')} for item in query_response["hits"]["hits"]]
    else:
        result_arr = [ {'idx':item['_source'].get('idx',0),'doc_category':item['_source']['doc_category'],'doc_classify':item['_source'].get('doc_classify'),'doc_title':item['_source']['doc_title'],'id':item['_id'],'doc':item['_source']['content'], 'doc_type': item['_source']['doc_type'], 'score': item['_score'],'doc_author': item['_source']['doc_author'], 'doc_meta': item['_source'].get('doc_meta','')} for item in query_response["hits"]["hits"]]
    return result_arr

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
       
     ## kkn前置检索FAQ,，如果query非常相似，则返回作为cache
    def knn_quick_prefetch(self, query_input: str, prefetch_threshold:float) -> List[Any]:
        start = time.time()
        query_embedding = get_vector_by_sm_endpoint(query_input, sm_client, self.embedding_model_endpoint)
        elpase_time = time.time() - start
        logger.info(f'knn_quick_prefetch, running time of get embeddings : {elpase_time:.3f}s')
        aos_client = OpenSearch(
                hosts=[{'host': self.aos_endpoint, 'port': 443}],
                http_auth = awsauth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection
            )
        start = time.time()
        opensearch_knn_respose = search_using_aos_knn(aos_client,query_embedding[0], self.aos_index,size=3)
        elpase_time = time.time() - start
        logger.info(f'runing time of quick_knn_fetch : {elpase_time:.3f}s')
        filter_knn_result = [item for item in opensearch_knn_respose if (item['score'] > prefetch_threshold and item['doc_type'] == 'Question')]
        if len(filter_knn_result) :
            filter_knn_result.sort(key=lambda x:x['score'])
        return filter_knn_result

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError
    
    def get_relevant_documents(self, query_input: str) -> List[Document]:
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

    def search_paragraph_neighbours(self,client, idx, doc_title,doc_category,doc_type, neighbor_cnt=1):
        query ={
            "query":{
                "bool": {
                "must": [
                    {
                    "terms": {
                        "idx": [i for i in range(idx-neighbor_cnt,idx+neighbor_cnt+1)]
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
    def rerank(self, query_input: str, docs: List[Any], sm_client, rerank_endpoint:str):
        inputs = [query_input]*len(docs)
        response_model = sm_client.invoke_endpoint(
            EndpointName=rerank_endpoint,
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
        return scores if isinstance(scores, list) else [scores]
    
    def de_duplicate(self, docs):
        unique_ids = set()
        nodup = []
        for item in docs:
            doc_hash = hashlib.md5(str(item['doc']).encode('utf-8')).hexdigest()
            if doc_hash not in unique_ids:
                nodup.append(item)
                unique_ids.add(doc_hash)
        return nodup

    def get_relevant_documents_from_bedrock(self, knowledge_base_id:str, query_input:str, top_k:int, rerank_endpoint=None, rerank_threshold=-2.0):
        BEDROCK_RETRIEVE_LIMIT=1000
        response = knowledgebase_client.retrieve(
            knowledgeBaseId=knowledge_base_id,
            retrievalQuery={
                'text': query_input[:BEDROCK_RETRIEVE_LIMIT]
            },
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': TOP_K,
                    'overrideSearchType': 'HYBRID',
                }
            },
        )

        def remove_s3_prefix(s3_path):
            return '/'.join(s3_path.split('/', 3)[3:])

        all_docs = [ {'idx': 1, 'rank_score':0, 'doc_classify':'-', 'doc_author':'-', 'doc_category': '-', 'doc_type':'Paragraph', 'doc':item['content']['text'],'score':item['score'], 'doc_title': remove_s3_prefix(item['location']['s3Location']['uri']) } for item in response['retrievalResults']]
        if rerank_endpoint:
            if all_docs:
                scores = self.rerank(query_input, all_docs, sm_client, rerank_endpoint)
                sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=False)
                recall_knowledge = [{**all_docs[idx],'rank_score':scores[idx] } for idx in sorted_indices[-top_k:] ]
                recall_knowledge = [item for item in recall_knowledge if item['rank_score'] >= rerank_threshold]
                return recall_knowledge
        
        return all_docs
    
    def get_websearch_documents(self, query_input: str) -> list:
        # 使用agent方式速度比较慢，直接改成调用search api
        all_docs = web_search(query=query_input)
        logger.info(f'all_docs:{all_docs}')
        recall_knowledge = [{'doc_title':item['title'],'doc':item['title']+'\n'+item['snippet'],
                             'doc_classify':'web_search','doc_type':'web_search','score':0.8,'doc_author':item['link']} for item in all_docs]
        return recall_knowledge
    
    def get_relevant_documents_custom(self, query_input: str, channel_return_cnt:int, top_k:int, knn_threshold:float, bm25_threshold:float, web_search_threshold:float, use_search:bool, rerank_endpoint:str=None, rerank_threshold:float=-2.0):
        # global BM25_QD_THRESHOLD_HARD_REFUSE, BM25_QD_THRESHOLD_SOFT_REFUSE
        # global KNN_QQ_THRESHOLD_HARD_REFUSE, KNN_QQ_THRESHOLD_SOFT_REFUSE,WEBSEARCH_THRESHOLD
        # global KNN_QD_THRESHOLD_HARD_REFUSE, KNN_QD_THRESHOLD_SOFT_REFUSE
        start = time.time()
        query_embedding = get_vector_by_sm_endpoint(query_input, sm_client, self.embedding_model_endpoint)
        elpase_time = time.time() - start
        logger.info(f'running time of get embeddings : {elpase_time:.3f}s')
        aos_client = OpenSearch(
                hosts=[{'host': self.aos_endpoint, 'port': 443}],
                http_auth = awsauth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection
            )
        start = time.time()
        opensearch_knn_respose = search_using_aos_knn(aos_client,query_embedding[0], self.aos_index,size=channel_return_cnt)
        elpase_time = time.time() - start
        logger.info(f'runing time of opensearch_knn : {elpase_time}s seconds')
        
        # 4. get AOS invertedIndex recall
        start = time.time()
        opensearch_query_response = aos_search(aos_client, self.aos_index, "doc", query_input,size=channel_return_cnt)
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
                knn_unique_ids = set()
                for item in opensearch_knn_respose:
                    doc_hash = hashlib.md5(str(item['doc']).encode('utf-8')).hexdigest()
                    if doc_hash not in knn_unique_ids:
                        opensearch_knn_nodup.append(item)
                        knn_unique_ids.add(doc_hash)
                
                opensearch_bm25_nodup = []
                bm25_unique_ids = set()
                for item in opensearch_query_response:
                    doc_hash = hashlib.md5(str(item['doc']).encode('utf-8')).hexdigest()
                    if doc_hash not in bm25_unique_ids:
                        opensearch_bm25_nodup.append(item)
                        doc_hash = hashlib.md5(str(item['doc']).encode('utf-8')).hexdigest()
                        bm25_unique_ids.add(doc_hash)

                opensearch_knn_nodup.sort(key=lambda x: x['score'])
                opensearch_bm25_nodup.sort(key=lambda x: x['score'])
                logger.info(f'opensearch_knn_nodup count:{len(opensearch_knn_nodup)}')
                logger.info(f'opensearch_bm25_nodup count:{len(opensearch_bm25_nodup)}')
                
                half_topk = math.ceil(topk/2) 
    
                kg_combine_result = [ item for item in opensearch_knn_nodup[-1*half_topk:]]

                doc_hash = set([ hashlib.md5(str(item['doc']).encode('utf-8')).hexdigest() for item in kg_combine_result ])

                for item in opensearch_bm25_nodup:
                    if len(kg_combine_result) >= topk:
                        break
                        
                    if hashlib.md5(str(item['doc']).encode('utf-8')).hexdigest() not in doc_hash:
                        kg_combine_result.append(item)
                        logger.info(f'kg_combine_result.append')
                    else:
                        continue

                return kg_combine_result
            
            ret_content = get_topk_items(opensearch_knn_respose, opensearch_query_response, top_k)
            logger.info(f'get_topk_items:{len(ret_content)}')
            return ret_content
        
        filter_knn_result = [ item for item in opensearch_knn_respose if item['score'] > knn_threshold ]
        filter_inverted_result = [ item for item in opensearch_query_response if item['score'] > bm25_threshold ]

        recall_knowledge = []
        ##是否使用rerank
        if rerank_endpoint:
            all_docs = filter_knn_result+filter_inverted_result

            ###to do 去重
            all_docs = self.de_duplicate(all_docs)
            if all_docs:
                scores = self.rerank(query_input, all_docs,sm_client,rerank_endpoint)
                ##sort by scores
                sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=False)
                recall_knowledge = [{**all_docs[idx],'rank_score':scores[idx] } for idx in sorted_indices[-TOP_K:] ] 
                
                ## 引入web search结果重新排序
                if max(scores) < web_search_threshold and use_search:
                    web_knowledge = self.get_websearch_documents(query_input)
                    if web_knowledge:
                        search_scores = self.rerank(query_input, web_knowledge,sm_client,rerank_endpoint)
                        sorted_indices = sorted(range(len(search_scores)), key=lambda i: search_scores[i], reverse=False)
                        
                        ## 过滤websearch结果
                        sorted_web_knowledge = [{**web_knowledge[idx],'rank_score':search_scores[idx] } for idx in sorted_indices if search_scores[idx]>=WEBSEARCH_THRESHOLD] 
                        ## 前面返回的是snippet内容，可以对结果继续用爬虫抓取完整内容
                        sorted_web_knowledge = add_webpage_content(sorted_web_knowledge)
                        
                        #添加到原有的知识里,并过滤到原来知识中的低分item
                        recall_knowledge += sorted_web_knowledge
                        #recall_knowledge = [item for item in  recall_knowledge if item['rank_score'] >= RERANK_THRESHOLD]
            elif use_search:
                ##如果没有找到知识，则直接搜索
                web_knowledge = self.get_websearch_documents(query_input)
                if web_knowledge:
                    search_scores = self.rerank(query_input, web_knowledge,sm_client,rerank_endpoint)
                    sorted_indices = sorted(range(len(search_scores)), key=lambda i: search_scores[i], reverse=False)
                    sorted_web_knowledge = [{**web_knowledge[idx],'rank_score':search_scores[idx] } for idx in sorted_indices if search_scores[idx]>=WEBSEARCH_THRESHOLD]
                    ## 前面返回的是snippet内容，可以对结果继续用爬虫抓取完整内容
                    recall_knowledge = add_webpage_content(sorted_web_knowledge)

            #filter unrelevant knowledge by rerank score
            recall_knowledge = [item for item in  recall_knowledge if item['rank_score'] >= rerank_threshold]
        else:
            logger.info(f'filter_knn_result count:{len(filter_knn_result)}')
            logger.info(f'filter_inverted_result count:{len(filter_inverted_result)}')
            recall_knowledge = combine_recalls(filter_knn_result, filter_inverted_result)
            recall_knowledge = [{**doc,'rank_score':0 } for doc in recall_knowledge]

        ##如果是段落类型，添加临近doc
        recall_knowledge = self.add_neighbours_doc(aos_client,recall_knowledge)

        return recall_knowledge,opensearch_knn_respose,opensearch_query_response
