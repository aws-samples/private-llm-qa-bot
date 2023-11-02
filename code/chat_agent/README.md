### 实现逻辑：
通过 prompt 模版实现 Agent功能，当前仅实现一个工具Agent，暂时未做控制 Agent


### 调用方法

```python
# call this lambda from another lambda
from boto3 import client as boto3_client
lambda_client = boto3_client('lambda')

def lambda_handler(event, context):
  	question = event['prompt'] 
    msg = {
      "params": {
        "query": query
      },
      "use_bedrock" : use_bedrock,
      "llm_model_name" : "anthropic.claude-v2"
    }
    response = lambda_client.invoke(FunctionName="Chat_Agent",
                                           InvocationType='RequestResponse',
                                           Payload=json.dumps(msg))
    
```


### 测试case

```json
#1
{
  "params": {
    "query": "Sagemaker相关问题应该联系谁？"  
  },
  "use_bedrock" : "True"
}

#2
{
  "params": {
    "query": "quicksight的GTMS是谁？",
  },
  "use_bedrock" : "True",
}

#3
{
  "params": {
    "query": "AIML北区的Sales是谁？",
  },
  "use_bedrock" : "True",
}

#4
{
  "params": {
    "query": "Emr相关问题应该联系谁？",
  },
  "use_bedrock" : "True",
}

#5
{  "params": {
    "query": "数据治理的GTMS是谁？",
  },
  "use_bedrock" : "True",
}

#6
{  "params": {
    "query": "aws head of sso 是谁",
  },
  "use_bedrock" : "True",
}

#7
{  "params": {
    "query": "aws sso大老板是谁",
  },
  "use_bedrock" : "True",
}

#8
{  "params": {
    "query": "Yinuo 负责AWS 什么服务",
  },
  "use_bedrock" : "True",
}

#9
{  "params": {
    "query": "Goden Yao 负责哪些服务",
  },
  "use_bedrock" : "True",
}

#10
{  "params": {
    "query": "Azure Competition 是谁负责",
  },
  "use_bedrock" : "True",
}

```


### TODO
1. Tool Agent 实现
2. Contorl Agent (React)

### 优化手段