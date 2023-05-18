from typing import Dict, List
# check https://github.dev/hwchase17/langchain for detailed class implementation
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.llms.sagemaker_endpoint import ContentHandlerBase
import json

class ContentHandler(ContentHandlerBase):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, inputs: list[str], model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"inputs": inputs, **model_kwargs})
        return input_str.encode('utf-8')

    def transform_output(self, output: bytes) -> List[List[float]]:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["vectors"]

content_handler = ContentHandler()

# make sure create endpoint first following instructions in embedding-model.ipynb
embeddings = SagemakerEndpointEmbeddings(
    endpoint_name="huggingface-pytorch-inference-2023-05-12-04-09-08-395", 
    region_name="us-west-2", 
    content_handler=content_handler
)

# generate embedding to allow user input simple query or documents
def generate_embedding(documents: List[str]) -> List[float]:
    return embeddings.embed_documents(documents)

# main entry to test the embedding
if __name__ == "__main__":
    documents = [
        "I like to eat apples",
        "I like to eat bananas",
        "I like to eat oranges",
        "I like to eat pears",
        "I like to eat peaches",
    ]
    embedding = generate_embedding(documents)
    print(embedding)