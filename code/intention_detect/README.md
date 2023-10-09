### 实现逻辑：

通过向量模型召回之前的一些意图样本作为fewshot，如果召回回来的fewshot意图都是一致的，那么直接返回该意图。 如果不一致，让大语言模型生成本次问句的意图。该意图应该在召回意图的范围内，如果不在范围内，则认为意图识别失败，返回Unknown.



### 准备工作：

1. 需要手动构建index

   ```
   PUT chatbot-example-index
   {
       "settings" : {
           "index":{
               "number_of_shards" : 1,
               "number_of_replicas" : 0,
               "knn": "true",
               "knn.algo_param.ef_search": 32
           }
       },
       "mappings": {
           "properties": {
               "publish_date" : {
                   "type": "date",
                   "format": "yyyy-MM-dd HH:mm:ss"
               },
               "intention" : {
                   "type" : "keyword"
               },
               "query": {
                   "type": "text",
                   "analyzer": "ik_max_word",
                   "search_analyzer": "ik_smart"
               },
               "reply": {
                   "type": "text"
               },
               "embedding": {
                   "type": "knn_vector",
                   "dimension": 1024,
                   "method": {
                       "name": "hnsw",
                       "space_type": "cosinesimil",
                       "engine": "nmslib",
                       "parameters": {
                           "ef_construction": 512,
                           "m": 32
                       }
                   }            
               }
           }
       }
   }
   ```

2. 通过前端注入样本，样本文件以.example为后缀，可以参见$repo/docs/intentions/目录中的文件



### 调用方法

```python
# call this lambda from another lambda
from boto3 import client as boto3_client
lambda_client = boto3_client('lambda')

def lambda_handler(event, context):
  	question = event['prompt'] #"DynamoDB怎么计费"
    msg = {"fewshot_cnt":5, "query": question }
    invoke_response = lambda_client.invoke(FunctionName="QAChatDeployStack-lambdaintention**",
                                           InvocationType='Event',
                                           Payload=json.dumps(msg))
    
```



### 测试case

```json
#1
{
  "fewshot_cnt": 5,
  "query": "DynamoDB怎么计费",
  "use_bedrock" : "True"
}

#2
{
  "fewshot_cnt": 5,
  "query": "AWS Control Tower怎么用？",
  "use_bedrock" : "True"
}

#3
{
  "fewshot_cnt": 5,
  "query": "g5.2xlarge单价是多少？",
  "use_bedrock" : "True"
}

#4
{
  "fewshot_cnt": 5,
  "query": "AWS 账户怎么加入到Organization？",
  "use_bedrock" : "True"
}

#5
{
  "fewshot_cnt": 5,
  "query": "想出去玩吗",
  "use_bedrock" : "True"
}
```



### 优化手段

1. 如果识别失败，可以添加更多的例子到OpenSearch的chatbot-example-index
2. 如果识别速度过慢，可以训练小模型(bert) 去分类