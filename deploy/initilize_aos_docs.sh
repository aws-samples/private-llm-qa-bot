#!/bin/bash

# Check if jq is installed
if ! command -v jq &> /dev/null ;then
    echo "jq could not be found. Please install jq [sudo yum install jq or sudo apt install jq] at first." 
    exit 1
fi

if [ "$#" -lt 1 ]; then
    echo "usage: $0 [region-name]"
    exit 1
fi

region=$1

stack_name="QAChatDeployStack"
output=$(aws cloudformation describe-stacks --stack-name "$stack_name" --region "$region" --query 'Stacks[0].Outputs[?OutputKey==`UPLOADBUCKET`].{OutputValue: OutputValue}' --output json)

bucketname=$(echo "$output" | jq -r '.[].OutputValue')
echo $output_values3
kd_path="s3://$bucketname/ai-content/init_docs/"

aws s3 cp ../docs/intentions $kd_path --recursive
aws s3 cp ../docs/aws_cleanroom.faq $kd_path
aws s3 cp ../docs/aws_emr.faq $kd_path
aws s3 cp ../docs/aws_msk.faq $kd_path
aws s3 cp ../docs/ask_user_faq.xlsx $kd_path