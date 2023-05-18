import langchain as lc
import requests
from langchain.vectorstores import OpenSearchVectorSearch
from embedding import generate_embedding

# Define LLM API endpoint 
LLM_URL = ""

# OpenSearch vector DB parameters 
OPENSEARCH_HOST = "your-opensearch-host" 
OPENSEARCH_PORT = 9200
OPENSEARCH_INDEX = "langchain"

# Initialize the VectorDB with OpenSearch params
# db = VectorDB(OPENSEARCH_HOST, OPENSEARCH_PORT, OPENSEARCH_INDEX)
db = 'your-vector-db'

# Define a function to get a response from your LLM API 
def query_llm(context):
    data = {
        'input': context,
        'stream': False
    }
    response = requests.post(LLM_URL, json=data)
    return response.json()

# Define a message generator function using langchain 
def generate_message(context, db):
    # Check if we have a matching context in our vector DB
    results = db.query(context) 
    
    if results: 
        # Get the best matching response from the DB 
        response = results[0]["response"]  
    else:
        # No match, generate a new response from the LLM API
        response = query_llm(context)  
        
        # Store the new context-response pair in the vector DB 
        db.add(context, response)   
        
    return response

# Function to get a response to a user message 
def respond(user_message, db):
    # Get the current conversation context
    context = db.get_context() 
    
    # Append the new user message to the context
    context += f' {user_message}'
    
    # Generate a response 
    response = generate_message(context, db)  
    
    # Update the context with the full conversation 
    db.set_context(context + f' {response}')
    
    # Return the response
    return response  

# main function
def main():
    # get user input and respond with a message, then continue until user says "bye"
    while True:
        user_message = input("Enter a message: ")
        if user_message == "bye":
            break
        else:
            # response = respond(user_message, db)
            # use query_llm() instead of respond() to get a response from the LLM API
            response = query_llm(user_message)
            print(response)

# run main function
if __name__ == "__main__":
    main()
