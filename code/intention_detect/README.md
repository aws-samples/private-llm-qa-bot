### 实现逻辑：

通过向量模型召回之前的一些意图样本作为fewshot，如果召回回来的fewshot意图都是一致的，那么直接返回该意图。 如果不一致，让大语言模型生成本次问句的意图。该意图应该在召回意图的范围内，如果不在范围内，则认为意图识别失败，返回Unknown.



### 准备工作：

1. 需要准备好对应的index(按照部署文档已经自动创建)

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
               "detection" : {
                   "type" : "keyword"
               },
               "query": {
                   "type": "text",
                   "analyzer": "ik_max_word",
                   "search_analyzer": "ik_smart"
               },
               "api_schema": {
                   "type": "keyword"
               },
               "doc_title": {
                   "type": "keyword"
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
    msg = {"fewshot_cnt":5, "query": question, "use_bedrock" : "True" }
    invoke_response = lambda_client.invoke(FunctionName="Detect_Intention",
                                           InvocationType='RequestResponse',
                                           Payload=json.dumps(msg))
    
```

### 集成方法
1. 在Lambda(Ask_Assistant)的环境变量中添加一个变量'intention_list'。以逗号分隔的字符串的形式，把除了默认包含的'闲聊'和'知识问答'以外的所有可能的意图枚举出来，如'intention_list' ->'Service角色查询,服务是否可用查询,价格咨询'
2. 目前支持bedrock-claude模型和SageMaker私有化部署的模型进行意图识别


### 测试方法
可以进入到'Detect_Intention' - Lambda中的Test选项卡中，使用下面json作为Event JSON来进行该模块的测试

#### 测试case
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

#6
{
  "fewshot_cnt": 5,
  "query": "bedrock国内可用吗？",
  "use_bedrock" : "True"
}

#7
{
  "fewshot_cnt": 5,
  "query": "DataZone的GTMS是谁？",
  "use_bedrock" : "True"
}

#8
{
  "fewshot_cnt": 5,
  "query": "EMR serverless中国区能用吗",
  "use_bedrock" : "True"
}
```

### 优化手段

1. 如果识别不准，可以添加更多的示例到example文件，然后更新摄入到OpenSearch的chatbot-example-index中
2. 如果不满意识别速度，当example例子数量积累到一定成都，可以训练分类模型(bert)
