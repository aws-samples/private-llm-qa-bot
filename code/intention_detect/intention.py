from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.llms.sagemaker_endpoint import ContentHandlerBase
import json
import os

class ContentHandler(ContentHandlerBase):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, inputs: list[str], model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"inputs": inputs, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> List[List[float]]:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["vectors"]

class llmContentHandler(LLMContentHandler):
    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({'inputs': prompt,'history':[],**model_kwargs})
        return input_str.encode('utf-8')
    
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["outputs"]

def create_intention_prompt_templete(lang='zh'):
    if lang == 'zh':
        prompt_template_zh = """{system_role_prompt}，能够回答{role_user}的各种问题以及陪{role_user}聊天,如:{chat_history}\n\n{role_user}: {question}\n{role_bot}:"""

        PROMPT = PromptTemplate(
            template=prompt_template_zh, 
            partial_variables={'system_role_prompt':SYSTEM_ROLE_PROMPT},
            input_variables=['question','chat_history','role_bot','role_user']
        )
    return PROMPT

@handle_error
def lambda_handler(event, context):
    embedding_endpoint_name = os.environ.get('endpoint_name')
    region = os.environ.get('region')
    aos_endpoint = os.environ.get('aos_endpoint')
    index_name = os.environ.get('index_name')
    query = event.get('query')
    fewshot_cnt = event.get('fewshot_cnt')
    llm_model_endpoint = os.environ.get('llm_model_endpoint')

    content_handler = ContentHandler()

    embeddings = SagemakerEndpointEmbeddings(
        endpoint_name=embedding_endpoint_name,
        region_name=region,
        content_handler=content_handler
    )

    docsearch = OpenSearchVectorSearch(
        index_name=index_name,
        embedding_function=embeddings,
        opensearch_url="http://{}:443".format(aos_endpoint),
    )
    
    docs = docsearch.similarity_search(
        query=query, 
        k = fewshot_cnt,
        space_type="cosinesimil",
        vector_field="embedding",
        text_field="query"
    )

    parameters = {
        "max_length": 10,
        "temperature": 0.01,
    }

    llmcontent_handler = llmContentHandler()
    llm=SagemakerEndpoint(
            endpoint_name=llm_model_endpoint, 
            region_name=region, 
            model_kwargs={'parameters':parameters},
            content_handler=llmcontent_handler
        )

    prompt_template = create_intention_prompt_templete(lang='zh')
    llmchain = LLMChain(llm=llm, verbose=False, prompt=prompt_template)
    answer = llmchain.run({'fewshot':docs})
    
    return answer
