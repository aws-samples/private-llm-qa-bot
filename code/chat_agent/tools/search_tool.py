# from langchain.utilities import GoogleSerperAPIWrapper
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import initialize_agent,AgentType,Tool,AgentExecutor
from langchain.llms.bedrock import Bedrock
from langchain.llms.base import LLM
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from pydantic import BaseModel,ValidationInfo, field_validator, Field,ValidationError
import os
import boto3
import time
from langchain import hub
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import SelfAskOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import argparse

import dotenv
dotenv.load_dotenv() 

boto3_bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name='us-east-1'
        )
 
 
class GoogleSearchAgent():
    search_agent: AgentExecutor

    def __init__(self,llm:LLM):
        search= GoogleSearchAPIWrapper()
        tools = [Tool(
            name='Intermediate Answer',
            description='useful for when you need to ask with search',
            func=search.run,
        )]
        self.search_agent = initialize_agent(
            tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True,
            handle_parsing_errors=True)

    def run(self,query):
        return self.search_agent.run(query)
    
class GoogleSearchTool():
    tool:Tool
    topk:int = 10
    
    def __init__(self,top_k=10):  
        self.topk = top_k
        search = GoogleSearchAPIWrapper()
        def top_results(query):
            return search.results(query, self.topk)
        self.tool = Tool(
            name="Google Search Snippets",
            description="Search Google for recent results.",
            func=top_results,
        )
        
    def run(self,query):
        return self.tool.run(query)
    
def web_search(**args):
    tool = GoogleSearchTool(top_k=args.get('top_k',10))
    result = tool.run(args['query'])
    return [item for item in result if 'title' in item and 'link' in item and 'snippet' in item]

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="search tool"
    )
    parser.add_argument("--query", type=str,default='你对阿里云2023年的云栖大会有什么看法')
    args = parser.parse_args()
    parameters = {
            "max_tokens_to_sample": 2048,
            "temperature":0.1,
            "top_p":0.85
        }
    llm = Bedrock(model_id='anthropic.claude-v2:1', client=boto3_bedrock, model_kwargs=parameters)
    import json
    sm_client = boto3.client("sagemaker-runtime")
    cross_model_endpoint='bge-reranker-2023-12-12-01-51-25-864-endpoint'
    def rerank(query_input: str, docs: List[Any],sm_client,cross_model_endpoint):
        inputs = [query_input]*len(docs)
        response_model = sm_client.invoke_endpoint(
            EndpointName=cross_model_endpoint,
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
        # scores = [item[1] for item in json_obj['scores']]
        scores = json_obj['scores']
        return scores
    
    def get_websearch_documents( query_input: str) -> list:
        all_docs = web_search(query=query_input)
        print(all_docs)
        recall_knowledge = [{'doc_title':item['title'],'doc':item['title']+'\n'+item['snippet'],
                             'doc_classify':'web_search','doc_type':'web_search','score':0.0,'doc_author':item['link']} for item in all_docs]
        return recall_knowledge

    t1 = time.time()
    query = args.query
    web_knowledge = get_websearch_documents(query)
    print(web_knowledge)
    search_scores = rerank(query, web_knowledge,sm_client,cross_model_endpoint)
    sorted_indices = sorted(range(len(search_scores)), key=lambda i: search_scores[i], reverse=False)
    recall_knowledge = [{**web_knowledge[idx],'rank_score':search_scores[idx] } for idx in sorted_indices if search_scores[idx]>1  ] 

    context = '\n'.join([item['doc'] for item in recall_knowledge])
    print(f'cost time: {time.time()-t1}')
    print(recall_knowledge)
    # [print (score, item) for score, item in zip(search_scores,web_knowledge)]
    
    # CONTEXT_TEMPLATE = "请严格根据以下的内容，回答用户的问题，不要自由发挥 \n<content>{context}</content>\n\n Quesiont:{question}"
    # prompt = PromptTemplate(
    #             template=CONTEXT_TEMPLATE,
    #             input_variables=["context",'question']
    #         )
    # llmchain = LLMChain(llm=llm,verbose=False,prompt = prompt)

    # print(f'llm input:{CONTEXT_TEMPLATE.format(context=context,question=query)}')
    # answer = llmchain.run({'context':context, "question": query})
    # print(llm(query))
    # print(f'cost time: {time.time()-t1}')

