### 模块功能：
1. 封装llm，包括bedrock模型和sagemaker私有化部署的模型
2. 提供可用model_id的list
3. 提供模型调用的lambda

### 调用方法

```python
# call this lambda from another lambda
from boto3 import client as boto3_client
lambda_client = boto3_client('lambda')

# 1. Get all available Model ids
msg = {"method" : "GET_ALL_MODEL_IDS"} 
invoke_response = lambda_client.invoke(FunctionName="Invoke_Generator",
                                        InvocationType='RequestResponse',
                                        Payload=json.dumps(msg))

# 2. Invoke LLM
msg = {"prompt": "hello", "model_id": "anthropic.claude-v2", "method" : "INVOKE_LLM"}
invoke_response = lambda_client.invoke(FunctionName="Invoke_Generator",
                                        InvocationType='RequestResponse',
                                        Payload=json.dumps(msg))
    
```

### 测试方法
可以进入到'Invoke_Generator' - Lambda中的Test选项卡中，使用下面json作为Event JSON来进行该模块的测试