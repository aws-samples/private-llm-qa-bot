#!/bin/bash

if [ "$#" -lt 1 ]||[ "$#" -gt 2 ]; then
    echo "usage: $0 [region-name] [account-id (optional)]"
    exit 1
fi

region=$1

if [ ! -n "$2" ]; then
    account_id=`aws sts get-caller-identity --query "Account" --output text`
else
    account_id="$2"
fi

ts=`date +%y-%m-%d-%H-%M-%S`
unique_tag="$account_id-$ts"

cn_region=("cn-north-1","cn-northwest-1")
if [[ "${cn_region[@]}" =~ "$region" ]]; then
    arn="arn:aws-cn:"
else
    arn="arn:aws:"
fi


embedding_endpoint="${unique_tag}-embedding-endpoint"
llm_model_endpoint="${unique_tag}-llm-default-endpoint"
bucket="${unique_tag}-bucket"
main_fun_arn="${arn}lambda:${region}:${account_id}:function:Ask_Assistant"
token_key="${unique_tag}"

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
echo "llm_model_endpoint=" >> .env
echo "cross_model_endpoint=" >> .env
echo "UPLOAD_BUCKET=${bucket}" >> .env
echo "UPLOAD_OBJ_PREFIX=ai-content/" >> .env
echo "neighbors=1" >>.env
echo "TOP_K=4" >>.env
echo "MAIN_FUN_ARN=${main_fun_arn}" >>.env
echo "TOKEN_KEY=${token_key}" >>.env

