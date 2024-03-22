import boto3
import pandas as pd
import json
from io import StringIO
import pandas as pd
import argparse
from langchain.text_splitter import MarkdownTextSplitter

def construct_gen_summary_prompt(content):
    prompt = f"""Please give a summary for below content.
    <content>
    {content}
    <content>

    Please skip the preamble, go straight into the answer. Put the answer in <summary>
    """

    messages = [ {"role":"user", "content": prompt}, {"role":"assistant", "content" : "\n<summary>"}]

    input_body = {}
    input_body["anthropic_version"] = "bedrock-2023-05-31"
    input_body["messages"] = messages
    input_body["max_tokens"] = 1024
    input_body["stop_sequences"] = ['</summary>']

    body = json.dumps(input_body)

    return body

def construct_QA_list_prompt(content):
    examples = """{"question" : [ "question1", "question2", ...]}"""

    prompt = f"""You are interviewing a candidate for a customer service at a well-established game company. 

    The ideal candidate should have experience to answer players' questions. Below is the given document.

    <document>
    {content}
    </document>

    Please give a series of questions based on the document, for checking the candidates's mastery。Avoid yes/no questions or those with obvious answers. 

    Please put your output in Json format, for example:

    {examples}
    """

    messages = [ 
        {"role":"user", "content" : prompt },
        {"role":"assistant", "content": "{\"questions\" : ["}
    ]

    input_body = {}
    input_body["anthropic_version"] = "bedrock-2023-05-31"
    input_body["messages"] = messages
    input_body["system"] = "You are a interviewer, your task is to generate a series of questions for the interview based. The questions should be asked frequently in realistic scenario， Especially in Game Customer Service."
    input_body["max_tokens"] = 1024
    input_body["stop_sequences"] = [']']

    body = json.dumps(input_body)

    return body

def markdown_spliter(file_content):
    md_splitter = MarkdownTextSplitter( 
        chunk_size = 512,
        chunk_overlap  = 0,
    )
    
    results = []
    chunks = md_splitter.create_documents([ file_content ] )
    for chunk in chunks:
        results.append(chunk.page_content)

    return results

def call_bedrock_enhance(content, prompt_func):
    request_body = prompt_func(content)
    request_options = {
        "body": request_body,
        "modelId": 'anthropic.claude-3-sonnet-20240229-v1:0',
        "accept": "application/json",
        "contentType": "application/json",
    }

    response = boto3_bedrock.invoke_model(**request_options)

    body = response.get('body').read().decode('utf-8')

    body_dict = json.loads(body)

    summary = body_dict['content'][0].get("text")

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, default='106839800180-24-02-07-02-51-47-bucket', help='input bucket')
    parser.add_argument('--prefix', type=str, default='mihoyo-poc/enhanced_test/', help='input bucket')
    parser.add_argument('--output_path', type=str, default='./output', help='output path')
    parser.add_argument('--region_name', type=str, default='us-west-2', help='aws region')
    args = parser.parse_args()

    output_path = args.output_path
    region = args.region_name
    bucket_name = args.bucket
    prefix = args.prefix

    boto3_bedrock = boto3.client(
        service_name="bedrock-runtime",
        region_name=region
    )

    s3 = boto3.client('s3')

    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    text_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.md')]

    for file in text_files:
        try:
            output_filename = '__'.join(file.split('/')[2:])
            print(f"processing file: {file}")
            obj = s3.get_object(Bucket=bucket_name, Key=file)
            content = obj['Body'].read().decode('utf-8')

            enhanced_data = []

            paragraphs = markdown_spliter(content)
            enhanced_data = [ [p, content, 'Nochunk-Paragraph'] for p in paragraphs ]
            
            enhanced_data.append([content, content, 'Nochunk-Page'])

            summary = call_bedrock_enhance(content, construct_gen_summary_prompt)
            enhanced_data.append([summary, content, 'Nochunk-Summary'])

            qa_list_str = call_bedrock_enhance(content, construct_QA_list_prompt)
            qa_list = [ [q.replace('"','').replace('\n','').strip(), content, 'Nochunk-Question' ] for q in qa_list_str.split('",') ]
            enhanced_data.extend(qa_list)

            df = pd.DataFrame(enhanced_data, columns=['Question', 'Answer', 'doc_type'])
            df.to_excel(f'{output_path}/{output_filename}.xlsx', index=False)
        except Exception as e:
            print(str(e))
