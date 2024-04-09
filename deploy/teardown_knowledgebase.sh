#!/bin/bash

OPENSEARCH_ENDPOINT="$1"
DIMENSION="$2"
COMPANY="${3:-default}"

# 创建chatbot-index索引
#echo $payload1 
echo "delete existed index[chatbot-index-$COMPANY] of opensearch."
curl -XDELETE "$OPENSEARCH_ENDPOINT/chatbot-index-$COMPANY" -H "Content-Type: application/json"
echo 
echo

# 创建chatbot-index索引
# echo $payload2
echo "delete existed index[chatbot-index-example-$COMPANY] of opensearch."
curl -XDELETE "$OPENSEARCH_ENDPOINT/chatbot-example-index-$COMPANY" -H "Content-Type: application/json"
echo 