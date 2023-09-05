#!/bin/bash

region=$1

if [ ! -n "$2" ]; then
    account_id=`aws sts get-caller-identity --query "Account" --output text`
else
    account_id="$2"
fi

ts=`date +%y-%m-%d-%H-%M-%S`
unique_tag="$account_id-$ts"

embedding_endpoint="${unique_tag}-embedding-endpoint"
llm_default_endpoint="${unique_tag}-llm-default-endpoint"
llm_bloomz_endpoint="${unique_tag}-llm-bloomz-endpoint"
llm_chatglm_endpoint="${unique_tag}-llm-chatglm-endpoint"
llm_llama_endpoint="${unique_tag}-llm-llama-endpoint"
llm_alpaca_endpoint="${unique_tag}-llm-alpaca-endpoint"
llm_vicuna_endpoint="${unique_tag}-llm-vicuna-endpoint"
llm_chatglm_stream_endpoint="${unique_tag}-llm-chatglm-stream-endpoint"
bucket="${account_id}-knowledge-bucket"
wss_resourceArn="arn:aws:execute-api:us-east-2:946277762357:3g36ob2mc2/*/*/@connections/*"
wss_apiId="3g36ob2mc2"
echo "CDK_DEFAULT_ACCOUNT=${account_id}" > .env
echo "CDK_DEFAULT_REGION=${region}" >> .env
echo "existing_vpc_id=optional" >> .env
echo "Kendra_index_id=f36f5962-4ca8-4a65-9c60-5b813e5f46bc" >> .env
echo "Kendra_result_num=3" >> .env
echo "aos_index=chatbot-index" >> .env
echo "aos_knn_field=embedding" >> .env
echo "aos_results=3" >> .env
echo "aos_existing_endpoint=optional" >> .env
echo "embedding_endpoint=${embedding_endpoint}" >> .env
echo "llm_default_endpoint=${llm_default_endpoint}" >> .env
echo "llm_bloomz_endpoint=${llm_bloomz_endpoint}" >> .env
echo "llm_chatglm_endpoint=${llm_chatglm_endpoint}" >> .env
echo "llm_llama_endpoint=${llm_llama_endpoint}" >> .env
echo "llm_alpaca_endpoint=${llm_alpaca_endpoint}" >> .env
echo "llm_vicuna_endpoint=${llm_vicuna_endpoint}" >> .env
echo "llm_chatglm_stream_endpoint=${llm_chatglm_stream_endpoint}" >> .env
echo "llm_visualglm_endpoint=" >> .env
echo "llm_other_stream_endpoint=" >> .env
echo "knn_threshold=0.5" >> .env
echo "inverted_theshold=5.0" >> .env
echo "UPLOAD_BUCKET=${bucket}" >> .env
echo "UPLOAD_OBJ_PREFIX=ai-content/" >> .env
echo "wss_apiId=${wss_apiId}" >> .env
echo "wss_resourceArn=${wss_resourceArn}" >> .env
echo "neighbors=1" >>.env
echo "TOP_K=4" >>.env

