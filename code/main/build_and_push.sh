#!/bin/bash
set -v
set -e

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the region name. 

if [ "$#" -lt 1 ]||[ "$#" -gt 2 ]; then
    echo "usage: $0 [region-name] [profile-name (optional)]"
    exit 1
fi

region=$1
suffix="com"

cn_region=("cn-north-1","cn-northwest-1")

if [[ "${cn_region[@]}" =~ "$region" ]]; then
    suffix="com.cn"
fi

if [ "$#" -eq 2 ]; then
    profile=$2
else
    profile="default"
fi


# Get the account number associated with the current IAM credentials
account=$(aws sts --profile $profile get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi

func_image=chatbot_qa_main_brain_func
funcimage_fullname=${account}.dkr.ecr.${region}.amazonaws.${suffix}/${func_image}:latest

# If the repository doesn't exist in ECR, create it.
aws --profile ${profile} ecr describe-repositories --repository-names "${func_image}" --region ${region} || aws --profile ${profile} ecr create-repository --repository-name "${func_image}" --region ${region}

if [ $? -ne 0 ]
then
    aws --profile ${profile} ecr create-repository --repository-name "${func_image}" --region ${region}
fi

# Get the login command from ECR and execute it directly
aws --profile ${profile} ecr get-login-password --region $region | docker login --username AWS --password-stdin $account.dkr.ecr.$region.amazonaws.${suffix}

aws --profile ${profile} ecr set-repository-policy \
    --repository-name "${func_image}" \
    --policy-text "file://ecr-policy.json" \
    --region ${region}

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

docker build -t ${func_image} -f Dockerfile . --platform=linux/amd64

docker tag ${func_image} ${funcimage_fullname}

docker push ${funcimage_fullname}