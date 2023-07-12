#!/usr/bin/env python
# coding: utf-8

from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth, helpers
import boto3
import random
import json
from awsglue.utils import getResolvedOptions
import sys
import hashlib
import datetime
import re
import os
import itertools
from bs4 import BeautifulSoup
from langchain.document_loaders import PDFMinerPDFasHTMLLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter

args = getResolvedOptions(sys.argv, ['bucket', 'object_key','AOS_ENDPOINT','REGION','EMB_MODEL_ENDPOINT'])
s3 = boto3.resource('s3')
bucket = args['bucket']
object_key = args['object_key']
QA_SEP = '=====' # args['qa_sep'] # 
EXAMPLE_SEP = '\n\n'
arg_chunk_size = 384

# EMB_MODEL_ENDPOINT = "st-paraphrase-mpnet-base-v2-2023-04-19-04-14-31-658-endpoint"
EMB_MODEL_ENDPOINT=args['EMB_MODEL_ENDPOINT']
smr_client = boto3.client("sagemaker-runtime")

# AOS_ENDPOINT = 'vpc-chatbot-knn-3qe6mdpowjf3cklpj5c4q2blou.us-east-1.es.amazonaws.com'
AOS_ENDPOINT = args['AOS_ENDPOINT']
REGION = args['REGION']
INDEX_NAME = 'chatbot-index'
EXAMPLE_INDEX_NAME = 'chatbot-example-index'

DOC_INDEX_TABLE= 'chatbot_doc_index'
dynamodb = boto3.client('dynamodb')



def get_embedding(smr_client, text_arrs, endpoint_name=EMB_MODEL_ENDPOINT):
    parameters = {
      #"early_stopping": True,
      #"length_penalty": 2.0,
      "max_new_tokens": 50,
      "temperature": 0,
      "min_length": 10,
      "no_repeat_ngram_size": 2,
    }

    response_model = smr_client.invoke_endpoint(
                EndpointName=endpoint_name,
                Body=json.dumps(
                {
                    "inputs": text_arrs,
                    "parameters": parameters
                }
                ),
                ContentType="application/json",
            )
    
    json_str = response_model['Body'].read().decode('utf8')
    json_obj = json.loads(json_str)
    embeddings = json_obj["sentence_embeddings"]
    
    return embeddings

def batch_generator(generator, batch_size):
    while True:
        batch = list(itertools.islice(generator, batch_size))
        if not batch:
            break
        yield batch

def iterate_paragraph(file_content, object_key,smr_client, index_name, endpoint_name):
    json_arr = json.loads(file_content)
    publish_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    doc_title = object_key

    for idx, json_item in enumerate(json_arr):
        header = ""
        if len(json_item['heading']) > 0:
            header = json_item['heading'][0]['heading']
            
        paragraph_content = json_item['content']

        if len(paragraph_content) > 1024:
            continue

        #whole paragraph embedding
        whole_paragraph_emb = get_embedding(smr_client, [paragraph_content,], endpoint_name)
        document = { "publish_date": publish_date, "idx":idx, "doc" : paragraph_content, "doc_type" : "Paragraph", "content" : paragraph_content, "doc_title": doc_title, "doc_category": "", "embedding" : whole_paragraph_emb[0]}
        yield {"_index": index_name, "_source": document, "_id": hashlib.md5(str(document).encode('utf-8')).hexdigest()}
        
        # for every sentence
        sentences = re.split('[。？?.！!]', paragraph_content)
        sentences = [ sent for sent in sentences if len(sent) > 2 ]
        sentences_count = len(sentences)
        print("Process {} sentences in one batch".format(sentences_count))
        start = 0 
        while start < sentences_count:
            sentence_slices = sentences[start:start+20]
            print("Process {}-{} sentences in one micro-batch".format(start, start+20))
            start += 20
            embeddings = get_embedding(smr_client, sentence_slices, endpoint_name)
            for sent_id, sent in enumerate(sentence_slices):
                document = { "publish_date": publish_date, "idx":idx, "doc" : sent, "doc_type" : "Sentence", "content" : paragraph_content, "doc_title": doc_title, "doc_category": "", "embedding" : embeddings[sent_id]}
                yield {"_index": index_name, "_source": document, "_id": hashlib.md5(str(document).encode('utf-8')).hexdigest()}

def iterate_QA(file_content, object_key,smr_client, index_name, endpoint_name):
    json_content = json.loads(file_content)
    json_arr = json_content["qa_list"]

    doc_title = object_key
    doc_category = json_content["doc_category"]
    questions = [ json_item['Question'] for json_item in json_arr ]
    answers = [ json_item['Answer'] for json_item in json_arr ]
    embeddings = get_embedding(smr_client, questions, endpoint_name)

    publish_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for i in range(len(embeddings)):
        document = { "publish_date": publish_date, "doc" : questions[i], "doc_type" : "Question", "content" : answers[i], "doc_title": doc_title, "doc_category": doc_category, "embedding" : embeddings[i]}
        yield {"_index": index_name, "_source": document, "_id": hashlib.md5(str(document).encode('utf-8')).hexdigest()}

def iterate_examples(file_content, object_key, smr_client, index_name, endpoint_name):
    json_arr = json.loads(file_content)

    it = iter(json_arr)
    example_batches = batch_generator(it, batch_size=3)
    publish_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for idx, batch in enumerate(example_batches):

        queries = [ item['query'] for item in batch ]
        intentions = [ item['intention'] for item in batch ]
        replies = [ item['reply'] for item in batch ]

        embeddings = get_embedding(smr_client, queries, endpoint_name)
        for i, query in enumerate(queries):
            print("query:")
            print(query)
            document = { "publish_date": publish_date, "intention" : intentions[i], "query" : queries[i], "reply" : replies[i], "embedding" : embeddings[i]}
            yield {"_index": index_name, "_source": document, "_id": hashlib.md5(str(document).encode('utf-8')).hexdigest()}
            
def link_header(semantic_snippets):
    heading_fonts_arr = [ item.metadata['heading_font'] for item in semantic_snippets ]
    heading_arr = [ item.metadata['heading'] for item in semantic_snippets ]

    def fontsize_mapping(heading_fonts_arr):
        heading_fonts_set = list(set(heading_fonts_arr))
        heading_fonts_set.sort(reverse=True)
        idxs = range(len(heading_fonts_set))
        font_idx_mapping = dict(zip(heading_fonts_set,idxs))
        return font_idx_mapping
        
    fontsize_dict = fontsize_mapping(heading_fonts_arr)

    snippet_arr = []
    for idx, snippet in enumerate(semantic_snippets):
        font_size = heading_fonts_arr[idx]
        heading_stack = []
        heading_info = {"font_size":heading_fonts_arr[idx], "heading":heading_arr[idx], "fontsize_idx" : fontsize_dict[font_size]}
        heading_stack.append(heading_info)
        for id in range(0,idx)[::-1]:
            if font_size < heading_fonts_arr[id]:
                font_size = heading_fonts_arr[id]
                heading_info = {"font_size":font_size, "heading":heading_arr[id], "fontsize_idx" : fontsize_dict[font_size]}
                heading_stack.append(heading_info)
            
        snippet_info = {
            "heading" : heading_stack,
            "content" : snippet.page_content
        }
        snippet_arr.append(snippet_info)
        
    json_arr = json.dumps(snippet_arr, ensure_ascii=False)
    return json_arr

def parse_pdf_to_json(file_content):
    soup = BeautifulSoup(file_content,'html.parser')
    content = soup.find_all('div')

    cur_fs = None
    cur_text = ''
    snippets = []   # first collect all snippets that have the same font size
    for c in content:
        sp = c.find('span')
        if not sp:
            continue
        st = sp.get('style')
        if not st:
            continue
        fs = re.findall('font-size:(\d+)px',st)
        if not fs:
            continue
        fs = int(fs[0])
        if not cur_fs:
            cur_fs = fs
        if fs == cur_fs:
            cur_text += c.text
        else:
            snippets.append((cur_text,cur_fs))
            cur_fs = fs
            cur_text = c.text
    snippets.append((cur_text,cur_fs))

    cur_idx = -1
    semantic_snippets = []
    # Assumption: headings have higher font size than their respective content
    for s in snippets:
        # if current snippet's font size > previous section's heading => it is a new heading
        if not semantic_snippets or s[1] > semantic_snippets[cur_idx].metadata['heading_font']:
            metadata={'heading':s[0], 'content_font': 0, 'heading_font': s[1]}
            #metadata.update(data.metadata)
            semantic_snippets.append(Document(page_content='',metadata=metadata))
            cur_idx += 1
            continue
        
        # if current snippet's font size <= previous section's content => content belongs to the same section (one can also create
        # a tree like structure for sub sections if needed but that may require some more thinking and may be data specific)
        if not semantic_snippets[cur_idx].metadata['content_font'] or s[1] <= semantic_snippets[cur_idx].metadata['content_font']:
            semantic_snippets[cur_idx].page_content += s[0]
            semantic_snippets[cur_idx].metadata['content_font'] = max(s[1], semantic_snippets[cur_idx].metadata['content_font'])
            continue
        
        # if current snippet's font size > previous section's content but less tha previous section's heading than also make a new 
        # section (e.g. title of a pdf will have the highest font size but we don't want it to subsume all sections)
        metadata={'heading':s[0], 'content_font': 0, 'heading_font': s[1]}
        #metadata.update(data.metadata)
        semantic_snippets.append(Document(page_content='',metadata=metadata))
        cur_idx += 1

    json_content = link_header(semantic_snippets)
    return json_content

def parse_faq_to_json(file_content):
    arr = file_content.split(QA_SEP)
    json_arr = []
    for item in arr:
        question, answer = item.strip().split("\n", 1)
        question = question.replace("Question: ", "")
        answer = answer.replace("Answer: ", "")
        obj = {
            "Question":question, "Answer":answer
        }
        json_arr.append(obj)

    qa_content = {
        "doc_title" : "",
        "doc_category" : "",
        "qa_list" : json_arr
    }
    
    json_content = json.dumps(qa_content, ensure_ascii=False)
    return json_content
    
def parse_txt_to_json(file_content):
    text_splitter = RecursiveCharacterTextSplitter( 
        chunk_size = arg_chunk_size,
        chunk_overlap  = 0,
    )
    
    results = []
    chunks = text_splitter.create_documents([ file_content ] )
    for chunk in chunks:
        snippet_info = {
            "heading" : [],
            "content" : chunk.page_content
        }
        results.append(snippet_info)

    json_content = json.dumps(results, ensure_ascii=False)
    return json_content

def parse_example_to_json(file_content):
    arr = file_content.split(EXAMPLE_SEP)
    json_arr = []

    for item in arr:
        elements = item.strip().split("\n")
        print("elements:")
        print(elements)
        obj = { element.split(":")[0] : element.split(":")[1] for element in elements }
        json_arr.append(obj)

    qa_content = {
        "example_list" : json_arr
    }
    
    json_content = json.dumps(qa_content, ensure_ascii=False)
    return json_content

def parse_html_to_json(html_docs):
    text_splitter = RecursiveCharacterTextSplitter( 
        chunk_size = arg_chunk_size,
        chunk_overlap  = 0,
    )

    results = []
    chunks = text_splitter.create_documents([ doc.page_content for doc in docs ] )
    for chunk in chunks:
        snippet_info = {
            "heading" : [],
            "content" : chunk.page_content
        }
        results.append(snippet_info)
    json_content = json.dumps(results, ensure_ascii=False)
    return json_content

def load_content_json_from_s3(bucket, object_key, content_type, credentials):
    if content_type == 'pdf':
        pdf_path=os.path.basename(object_key)
        s3_client=boto3.client('s3', region_name=REGION)
        s3_client.download_file(Bucket=bucket, Key=object_key, Filename=pdf_path)
        loader = PDFMinerPDFasHTMLLoader(pdf_path)
        file_content = loader.load()[0].page_content
        json_content = parse_pdf_to_json(file_content)
        return json_content
    else:
        obj = s3.Object(bucket,object_key)
        file_content = obj.get()['Body'].read().decode('utf-8').strip()
        
        if content_type == 'faq':
            json_content = parse_faq_to_json(file_content)
        elif content_type =='txt':
            json_content = parse_txt_to_json(file_content)
        elif content_type =='json':
            json_content = file_content
        elif content_type == 'example':
            json_content = file_content
        else:
            raise "unsupport content type...(pdf, faq, txt are supported.)"
        
        return json_content


def put_idx_to_ddb(filename,username,index_name,embedding_model):
    try:
        dynamodb.put_item(
            Item={
                'filename':{
                    'S':filename,
                },
                'username':{
                    'S':username,
                },
                'index_name':{
                    'S':index_name,
                },
                'embedding_model':{
                    'S':embedding_model,
                }
            },
            TableName = DOC_INDEX_TABLE,
        )
        print(f"Put filename:{filename} with embedding:{embedding_model} index_name:{index_name} by user:{username} to ddb success")
        return True
    except Exception as e:
        print(f"There was an error put filename:{filename} with embedding:{embedding_model} index_name:{index_name} to ddb: {str(e)}")
        return False 


def query_idx_from_ddb(filename,username,embedding_model):
    try:
        response = dynamodb.query(
            TableName=DOC_INDEX_TABLE,
            ExpressionAttributeValues={
                ':v1': {
                    'S': filename,
                },
                ':v2': {
                    'S': username,
                },
                ':v3': {
                    'S': embedding_model,
                },
            },
            KeyConditionExpression='filename = :v1 and username = :v2',
            ExpressionAttributeNames={"#e":"embedding_model"},
            FilterExpression='#e = :v3',
            ProjectionExpression='index_name'
        )
        if len(response['Items']):
            index_name = response['Items'][0]['index_name']['S'] 
        else:
            index_name = ''
        print (f"query filename:{filename} with embedding:{embedding_model} index_name:{index_name} from ddb")
        return index_name
    
    except Exception as e:
        print(f"There was an error an error query filename:{filename} index from ddb: {str(e)}")
        return ''

def get_idx_from_ddb(filename,embedding_model):
    try:
        response = dynamodb.get_item(
            Key={
            'filename':{
            'S':filename,
            },
            'embedding_model':{
            'S':embedding_model,
            },
            },
            TableName = DOC_INDEX_TABLE,
        )
        index_name = ''
        if response.get('Item'):
            index_name = response['Item']['index_name']['S']
            print (f"Get filename:{filename} with index_name:{index_name} from ddb")
        return index_name
    except Exception as e:
        print(f"There was an error get filename:{filename} with embedding:{embedding_model} index from ddb: {str(e)}")
        return ''
    
def WriteVecIndexToAOS(bucket, object_key, content_type, smr_client, aos_endpoint=AOS_ENDPOINT, region=REGION, index_name=INDEX_NAME):
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, region)
    # auth = ('xxxx', 'yyyy') master user/pwd
    # auth = (aos_master, aos_pwd)
    try:
        file_content = load_content_json_from_s3(bucket, object_key, content_type, credentials)
        # print("file_content:")
        # print(file_content)

        client = OpenSearch(
            hosts = [{'host': aos_endpoint, 'port': 443}],
            http_auth = auth,
            use_ssl = True,
            verify_certs = True,
            connection_class = RequestsHttpConnection
        )

        gen_aos_record_func = None
        if content_type == "faq":
            gen_aos_record_func = iterate_QA(file_content, object_key,smr_client, index_name, EMB_MODEL_ENDPOINT)
        elif content_type in ['txt', 'pdf', 'json']:
            gen_aos_record_func = iterate_paragraph(file_content,object_key, smr_client, index_name, EMB_MODEL_ENDPOINT)
        elif content_type in ['example']:
            gen_aos_record_func = iterate_examples(file_content,object_key, smr_client, index_name, EMB_MODEL_ENDPOINT)
        else:
            raise RuntimeError('No Such Content type supported') 

        response = helpers.bulk(client, gen_aos_record_func)
        return response
    except Exception as e:
        print(f"There was an error when ingest:{object_key} to aos cluster, Exception: {str(e)}")
        return ''    

def process_s3_uploaded_file(bucket, object_key):
    print("********** object_key : " + object_key)

    content_type = None
    index_name = INDEX_NAME
    if object_key.endswith(".faq"):
        print("********** pre-processing faq file")
        content_type = 'faq'
    elif object_key.endswith(".txt"):
        print("********** pre-processing text file")
        content_type = 'txt'
    elif object_key.endswith(".pdf"):
        print("********** pre-processing pdf file")
        content_type = 'pdf'
    elif object_key.endswith(".json"):
        print("********** pre-processing json file")
        content_type = 'json'
    elif object_key.endswith(".example"):
        print("********** pre-processing example file")
        content_type = 'example'
        index_name = EXAMPLE_INDEX_NAME
    else:
        raise "unsupport content type...(pdf, faq, txt are supported.)"
    
    #check if it is already built
    idx_name = get_idx_from_ddb(object_key,EMB_MODEL_ENDPOINT)
    if len(idx_name) > 0:
        print("doc file already exists")
        return
    
    response = WriteVecIndexToAOS(bucket, object_key, content_type, smr_client, index_name=index_name)
    print("response:")
    print(response)
    print("ingest {} chunk to AOS".format(response[0]))
    put_idx_to_ddb(filename=object_key,username='s3event',
                    index_name=index_name,
                        embedding_model=EMB_MODEL_ENDPOINT)

for s3_key in object_key.split(','):
    process_s3_uploaded_file(bucket, s3_key)