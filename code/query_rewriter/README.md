### 实现逻辑：

参考Langchain的prompt_template实现自己的query rewriter，通过lambda独立成一个接口供调用


### 调用方法

```python
# call this lambda from another lambda
from boto3 import client as boto3_client
lambda_client = boto3_client('lambda')

def lambda_handler(event, context):
  	question = event['prompt'] #"DynamoDB怎么计费"
    msg = {
      "params": {
        "history": ["有戴森的吹风机吗？","没有哦亲亲", "戴森都没有", "不好意思，看看其他品牌呢"],
        "query": question
      },
      "use_bedrock" : "True"
    }
    invoke_response = lambda_client.invoke(FunctionName="Detect_Intention",
                                           InvocationType='RequestResponse',
                                           Payload=json.dumps(msg))
    
```


### 测试case

```json
#1
{
  "params": {
    "history": ["有戴森的吹风机吗？","没有哦亲亲", "戴森都没有", "不好意思，看看其他品牌呢"],
    "query": "那有松下的吗？"  
  },
  "use_bedrock" : "True"
}

#2
{
  "params": {
    "history": ["你喜欢周杰伦吗", "我喜欢周杰伦"],
    "query": "你喜欢他哪首歌",
  },
  "use_bedrock" : "True",
  "role_a" : "user",
  "role_b" : "bot"
}
```


### 优化手段

1. 速度层面，使用IUR(Incomplete Utterance Rewrite)相关的小模型进行推理