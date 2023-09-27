#!/bin/bash
exec > /var/log/user-data.log 2>&1

# 输出日志消息
echo "Starting user data script..."
emb_model=$1
region=$2
aos_endpoint=$3
yum update -y
amazon-linux-extras install nginx1 -y
amazon-linux-extras install jq -y

echo "start pip3 installation"
pip3 install boto3
pip3 install huggingface-hub -Uqq
pip3 install -U sagemaker
pip3 install opensearch-py
echo "finish pip3 installation"

echo "enter /home/ec2-user"
cd /home/ec2-user

echo "execute init_aos_client.py"
curl -LJO https://raw.githubusercontent.com/aws-samples/private-llm-qa-bot/JAM20230915/deploy/init_aos_client.py
region=$(curl -s http://169.254.169.254/latest/dynamic/instance-identity/document | jq -r .region)
python3 init_aos_client.py $aos_endpoint $region
echo "finish execute init_aos_client.py"

echo "execute jam_deploy.py"
curl -LJO https://raw.githubusercontent.com/aws-samples/private-llm-qa-bot/JAM20230915/deploy/jam_deploy.py
python3 jam_deploy.py $emb_model $region
echo "finish execute jam_deploy.py"