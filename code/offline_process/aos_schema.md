## 1. 建立知识索引

```json
PUT chatbot-index
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
            "idx" : {
                "type": "integer"
            },
            "doc_type" : {
                "type" : "keyword"
            },
            "doc": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart"
            },
            "content": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart"
            },
            "doc_title": {
                "type": "keyword"
            },
            "doc_author": {
                "type": "keyword"
            },
            "doc_category": {
                "type": "keyword"
            },
            "embedding": {
                "type": "knn_vector",
                "dimension": 768,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",
                    "parameters": {
                        "ef_construction": 128,
                        "m": 16
                    }
                }            
            }
        }
    }
}
```

## 2. 建立意图识别example索引
```json
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
              "doc_title": {
                   "type": "keyword"
               },
               "embedding": {
                   "type": "knn_vector",
                   "dimension": 768,
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