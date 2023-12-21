import json
import os
import logging
from collections import Counter
from tqdm import tqdm
from langchain.prompts import PromptTemplate
from typing import Any, Dict, List, Union,Mapping, Optional, TypeVar, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.llms.bedrock import Bedrock
from botocore.exceptions import ClientError
import boto3
import math
import csv
import re
import argparse
from anthropic_bedrock import AnthropicBedrock
import docx
from datetime import datetime

import xml.etree.ElementTree as ET




num_size = 200
bedrock = AnthropicBedrock(
    aws_region="us-west-2",

)



boto3_bedrock = boto3.client(
        service_name="bedrock-runtime",
        region_name='us-west-2'
    )


prompt_template = """
  Please summarize below content in <content></content> to generate {num} standalone questions and correspondant answers, 
  response in xml tag <faq> for example:
<faq>
Q1:xxx
A1:xxx

Q2:xxx
A2:xxx
</faq>

<content>
{content}
</content>
"""

def extract_topics(content):
    pattern = r"<topic>(.*?)</topic>"
    matches = re.findall(pattern, content)
    topics = []
    for match in matches:
        topics.append(match)
    return topics
    
def extract_summary(content):
    pattern = r"<summary>(.*?)</summary>"
    matches = re.search(pattern, content,re.DOTALL)
    topics_and_contents = []
    if matches:
        content = matches.group()
        print(content)
        tree = ET.fromstring(content.strip())
        for chapter in tree.iter('chapter'):
            topics_and_contents.append((chapter.find("topic").text, chapter.find("content").text))
    return topics_and_contents


def parse_faq(faq_text):
    """
    解析 FAQ 文本并返回表格数据。

    Args:
    faq_text: FAQ 文本。

    Returns:
    表格数据。
    """

    # 匹配问题和答案的正则表达式
    pattern = r"(Q\d+):(.*)\n*(A\d+):(.*)"

    # 使用正则表达式匹配问题和答案
    matches = re.findall(pattern, faq_text)

    # 创建表格数据
    table_data = []
    for match in matches:
        table_data.append([match[1].strip(), match[3].strip(),'synthetic'])
    return table_data

def write_faq_pairs_to_csv(faq_pairs, csv_file):
    with open(csv_file, 'w', newline='',encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['Question', 'Answer','Author'])
        writer.writerows(faq_pairs)


def extract_faq(content: str):
    pattern = r"<faq>(.*?)</faq>"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        faq_text = match.group(1).strip()
        print(faq_text)
        return parse_faq(faq_text)
    else:
        return []


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = bedrock.count_tokens(string)  # 
    # print(f"num_tokens:{num_tokens}")
    return num_tokens


def generate_by_chap(doc_content:str):
    prompt_template1 = \
    """按章节总结<document>文档中的内容，并提取出章节对应的topic和和内容,生成xml格式的topic
    例如：
 <summary>
    <chapter>
        <topic>xxx</topic>
        <content>xxx</content>
    </chapter>
    <chapter>
        <topic>xxx</topic>
        <content>xxx</content>
    </chapter>
 </summary>
 
    <document>
    {document}
    </document>
    """
    PROMPT = PromptTemplate(
            template=prompt_template1, 
            input_variables=['document']
    )
    llmchain = get_bedrock_llm(PROMPT)
    outline_info = llmchain.run({"document":doc_content})
    print('outline_info:',outline_info)
    summary = extract_summary(outline_info)
    print('summary:',summary)
    
    prompt_template2= \
    """基于<document>提供的文档，生成一组FAQ(Frequently Ask Question), 大约按200字生成一对FAQ，要求生成的问题和答案是以对话语气，格式为json.
    response in xml tag <faq> for example:
    <faq>
    Q1:xxx
    A1:xxx
    
    Q2:xxx
    A2:xxx
    </faq>

    <document>
    title: {chapter}
    {document}
    </document>
    """
    PROMPT = PromptTemplate(
            template=prompt_template2, 
            input_variables=['chapter','document']
    )
    llmchain = get_bedrock_llm(PROMPT)
    qa_list = []
    for section_topic,chapter in tqdm(summary):
        print('section_topic:',section_topic)
        response = llmchain.run({'chapter':section_topic,"document":chapter})
        qa_list += extract_faq(response)
    return qa_list

    
    

    
 

def generate(file_name,llmchain:LLMChain,file_type:str):
    faq_table = []
    if file_type == 'csv':
        with open(file_name, 'r' ) as file:
            csv_data = file.readlines()
            reader = csv.reader(csv_data)
            header = next(reader)  # Skip the header row
            for item in reader:
                question, answer = item[0],item[1]
                content = f"Question:{question}\nAnswer:{answer}"
                faq_pairs = synth_faq(content,llmchain)
                faq_table += faq_pairs if faq_pairs else []
                # break #for test 
    elif file_type == 'docx':
        doc = docx.Document(file_name)
        text = ''
        for para in doc.paragraphs:
            text += '\n'+para.text
        faq_table = generate_by_chap(text)
        
    return faq_table


def synth_faq(content:str,llmchain:LLMChain):
    global num_size
    num_tokens = num_tokens_from_string(content)
    answer = llmchain.run({"content":content,"num":math.ceil(num_tokens/num_size)})
    # print(answer)
    return extract_faq(answer)


def get_bedrock_llm(prompt:PromptTemplate) ->LLMChain:
    parameters = {
        "max_tokens_to_sample": 8000,
        "stop_sequences": ["\n\nHuman:"],
        "temperature":0.1,
        "top_p":1
    }

    llm = Bedrock(model_id='anthropic.claude-v2:1', client=boto3_bedrock, model_kwargs=parameters)

    llmchain = LLMChain(llm=llm, verbose=False, prompt=prompt)
    return llmchain
    
    

def main():
    global num_size
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str,)
    parser.add_argument("--size", type=int, default=200)
    args = parser.parse_args()
    num_size = args.size  

    PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=['num','content']
    )
    
    llmchain = get_bedrock_llm(PROMPT)
    timestamp_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    input_file = args.file
    file_type = input_file.split(".")[-1]
    new_faq = generate(input_file,llmchain,file_type)
    outout_file = input_file.replace('.csv',"")+f'_{timestamp_str}_synth.csv'
    write_faq_pairs_to_csv(new_faq,outout_file)
    print(f'generated:{outout_file}')
    
if __name__ == '__main__':
    main()
#     xml_string = """here is
# <summary>
#     <chapter>
#         <topic>xxx</topic>
#         <content>xxx</content>
#     </chapter>
#     <chapter>
#         <topic>xxx</topic>
#         <content>xxx</content>
#     </chapter>
# </summary>
# """

#     topics_and_contents = extract_summary(xml_string)
#     print(topics_and_contents)

   
    

 
