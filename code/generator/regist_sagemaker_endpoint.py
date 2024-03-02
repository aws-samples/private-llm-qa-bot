from llm_manager import llm_endpoint_regist
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI For Regist SageMaker Endpoint')
    parser.add_argument('--endpoint', help='SageMaker Endpoint')
    parser.add_argument('--llm_model', help='Readable llm model name')
    args = parser.parse_args()

    endpoint_name = args.endpoint
    llm_model = args.llm_model
    
    llm_endpoint_regist(llm_model, endpoint_name)