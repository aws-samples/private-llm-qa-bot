import json
import os
import logging
from collections import Counter

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
        return None


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = bedrock.count_tokens(string)  # 
    # print(f"num_tokens:{num_tokens}")
    return num_tokens


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
        text_splitter = RecursiveCharacterTextSplitter(        
            chunk_size = 2000,
            chunk_overlap  = 50,
            length_function = num_tokens_from_string,
        )
        texts = text_splitter.split_text(text)
        print(f'-----total-chunks:{len(texts)}-----')
        for i,content in enumerate(texts):
            print(f'-----chunk:{i}--tokens:{num_tokens_from_string(content)}')
            print(content)
            faq_pairs = synth_faq(content,llmchain)
            faq_table += faq_pairs if faq_pairs else []
        
    return faq_table


def synth_faq(content:str,llmchain:LLMChain):
    num_tokens = num_tokens_from_string(content)
    answer = llmchain.run({"content":content,"num":math.ceil(num_tokens/100)})
    # print(answer)
    return extract_faq(answer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str,)
    args = parser.parse_args()
    parameters = {
        "max_tokens_to_sample": 4000,
        "stop_sequences": ["</output>"],
        "temperature":0.1,
        "top_p":1
    }

    llm = Bedrock(model_id='anthropic.claude-v2:1', client=boto3_bedrock, model_kwargs=parameters)

    PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=['num','content']
    )

    llmchain = LLMChain(llm=llm, verbose=False, prompt=PROMPT)

    input_file = args.file
    file_type = input_file.split(".")[-1]
    new_faq = generate(input_file,llmchain,file_type)
    outout_file = input_file.replace('.csv',"")+'_synth.csv'
    write_faq_pairs_to_csv(new_faq,outout_file)
    print(f'generated:{outout_file}')
    

 
