import boto3
import json
import requests
import time
from collections import defaultdict
from requests_aws4auth import AWS4Auth
import os
from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain.vectorstores import OpenSearchVectorSearch
from langchain import PromptTemplate, SagemakerEndpoint
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.llms.sagemaker_endpoint import ContentHandlerBase
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferWindowMemory
from langchain import LLMChain
from sagemaker.huggingface import HuggingFaceModel

source_includes = ["question","answer"]
runtime= boto3.client('runtime.sagemaker')
headers = { "Content-Type": "application/json" }

endpoint_name="pytorch-inference-2023-04-20-07-28-31-042"
region_name="us-west-2"

class ExtractContentHandler(ContentHandlerBase):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"ask": prompt})
        return input_str.encode('utf-8')

    def transform_output(self, output: bytes) -> str:
        model_predictions = json.loads(output.read().decode("utf-8"))
        generated_text = model_predictions["answer"]
        return generated_text
feature_extraction_handler = ExtractContentHandler()
feature_extraction_llm=SagemakerEndpoint(
    endpoint_name=endpoint_name,
    region_name=region_name,
    content_handler=feature_extraction_handler
)

########parse field search resule###############
# input:
#  r: AOS returned json
#  score: judgement score(default 0.75)
# return:
#  result : array of result(text, score)
#############################################
def parse_field_results(r,score=0.75):
    res = []
    result = []
    json_obj = json.loads(r)
    for i in range(len(json_obj['hits']['hits'])):
        h = json_obj['hits']['hits'][i]
        if (h['_score'])<score:
            #print("score=="+str(h['_score']))
            continue
        if h['_source']['question'] not in result :
            result.append(h['_source']['question'])
            res.append((h['_source']['question'] ,h['_score']))
    return res

########parse k-NN search resule###############
# input:
#  r: AOS returned json
# return:
#  result : array of topN text
#############################################
def parse_results(r):
    res = []
    result = []
    print(r)
    for i in range(len(r['hits']['hits'])):
        h = r['hits']['hits'][i]
        if h['_source']['question'] not in clean:
            result.append(h['_source']['question'])
            res.append('<第'+str(i+1)+'条信息>'+h['_source']['question'] + '。</第'+str(i+1)+'条信息>\n')
    print(res)
    return result

########get embedding vector by SM llm########
# input:
#  questions:question texts(list)
#  sm_client: sagemaker runtime client
#  sm_endpoint:Sagemaker embedding llm endpoint
# return:
#  result : vector of embeded text
#############################################
def get_vector_by_sm_endpoint(questions,sm_client,endpoint_name,parameters):
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



########get embedding vector by SM HF Model Wrapper########
# input:
#  questions:question texts(list)
#  sm_endpoint:Sagemaker embedding endpoint(huggingface-inference-text2vec-base-chinese-v1)
# return:
#  result : vector of embeded text
#############################################
def get_vector_by_hf_model(questions,endpoint_name):
    hfp = sagemaker.huggingface.model.HuggingFacePredictor(endpoint_name)
    return hfp.predict({'inputs':questions})[0][0][0]

########get embedding vector by lanchain vector search########
# input:
#  questions:question texts(list0
#  embedings:lanchain embeding models
# return:
#  result : vector of embeded text
#############################################################
def get_vector_by_lanchain(questions , embedings):
    doc_results = embeddings.embed_documents(questions)
    print(doc_results[0])
    return doc_results


########k-nn search by lanchain########
# input:
#  q:question text
#  vectorSearch: lanchain VectorSearch instance
# return:
#  result : k-NN search result
#############################################################
def search_using_lanchain(question, vectorSearch):
    docs = vectorSearch.similarity_search(query)
    return docs


########intension detect by aos knn########
# input:
#  query:question text
#  index:AOS k-NN index name
#  size: topN(for K-NN search)
#  hostname: aos endpoint
#  username: aos username
#  passwd: aos password
# return:
#  result : k-NN search result
#############################################################
def intension_detection_by_aos_knn(q_embedding, hostname, username,passwd, index, size):
    awsauth = (username, passwd)
    print(type(q_embedding))
    query = {
        "size": size,
        "query": {
            "knn": {
                "sentence_vector": {
                    "vector": q_embedding,
                    "k": size
                }
            }
        }
    }
    r = requests.post("https://"+hostname +"/"+ index + '/_search', auth=awsauth, headers=headers, json=query)
    return r.text

########intension detect by aos field########
# input:
#  question:query text
#  index:AOS k-NN index name
#  field:AOS field name(for document search)
#  hostname: aos endpoint
#  username: aos username
#  passwd: aos password
#  size: return size
# return:
#  result : field search result
#############################################################
def intension_detection_by_aos_field(question, hostname, username,passwd, index, field):
    awsauth = (username, passwd)
    query = {
        "size": size,
        "query": {
            "match": {
                field: question
            }
        }
    }
    r = requests.post("https://"+hostname +"/"+ index + '/_search', auth=awsauth, headers=headers, json=query)
    return r.text


########intension detect by llm ########
# input:
#  questions:question text
#  sm_client:sagemaker rutime client instance
#  endpoint_name:sagemaker llm hosting endpoint
#  parameters: inference parameters
# return:
#  result : llm prediction result
#############################################################
def intension_detection_by_sm_endpoint(questions,sm_client,endpoint_name):
    payload={"ask":questions+"\n"+
                   "问题：内容分类,以上属于闲聊还是专业问题？"}
    response_model = sm_client.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=json.dumps(
            payload
        ).encode("utf-8"),
        ContentType="application/json",
    )
    json_str = response_model['Body'].read().decode('utf8')
    json_obj = json.loads(json_str)
    result = json_obj['answer']
    return result


########k-nn by native AOS########
# input:
#  q_embedding:embeded question text(array)
#  index:AOS k-NN index name
#  hostname: aos endpoint
#  username: aos username
#  passwd: aos password
#  source_includes: fields to return
#  k: topN
# return:
#  result : k-NN search result
#############################################################
def search_using_aos_knn(q_embedding, hostname, username,passwd, index, source_includes, size):
    awsauth = (username, passwd)
    print(type(q_embedding))
    query = {
        "size": size,
        "query": {
            "knn": {
                "sentence_vector": {
                    "vector": q_embedding,
                    "k": size
                }
            }
        }
    }
    r = requests.post("https://"+hostname +"/"+ index + '/_search', auth=awsauth, headers=headers, json=query)
    return r.text



########k-nn ingestion by native AOS########
# input:
#  docs:ingestion source documents
#  index:AOS k-NN index name
#  hostname: aos endpoint
#  username: aos username
#  passwd: aos password
# return:
#  result : N/A
#############################################################
def k_nn_ingestion_by_aos(docs,index,hostname,username,passwd):
    auth = (username, passwd)
    search = OpenSearch(
        hosts = [{'host': hostname, 'port': 443}],
        ##http_auth = awsauth ,
        http_auth = auth ,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection
    )
    for doc in docs:
        vector_field = doc['sentence_vector']
        question_filed = doc['question']
        answer_field = doc['answer']
        document = { "question": question_filed, 'answer':answer_field, "sentence_vector": vector_field}
        search.index(index=index, body=document)


########k-nn ingestion by lanchain #########################
# input:
#  docs:ingestion source documents
#  vectorStore: lanchain AOS vectorStore instance
# return:
#  result : N/A
#############################################################
def k_nn_ingestion_by_lanchain(docs,vectorStore):
    for doc in docs:
        opensearch_vector_search.add_texts(docs,batch_size=10)


########topic extraction by lanchain sm endpoint #########################
# input:
#  docs: history QA list[(Q1,A1).(Q2,A2),(Q3,A3)]
#  k: output extracted words limit
#  llm: lanchain llm instance
# return:
#  result : output extracted features
#############################################################
def feature_extraction_by_lanchain(docs,k,sm_endpoint_nm,sm_region):
    #feature_extraction_handler = ExtractContentHandler()
    #feature_extraction_llm=SagemakerEndpoint(
    #    endpoint_name=sm_endpoint_nm,
    #    region_name=sm_region,
    #    content_handler=feature_extraction_handler
    #)
    global feature_extraction_llm
    if feature_extraction_llm is None:
        feature_extraction_llm=SagemakerEndpoint(
            endpoint_name=sm_endpoint_nm,
            region_name=sm_region,
            content_handler=feature_extraction_handler
        )
    payload = ""
    for doc in docs:
        question, answer = doc
        payload=payload+"Q:"+question+"\n"
        payload=payload+"A:"+answer+"\n"
    payload = payload+"问题：主题摘要,限制在"+str(k)+"个字内"
    extracted_feature_texts=feature_extraction_llm(payload)
    return extracted_feature_texts