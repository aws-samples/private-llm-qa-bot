- 代码介绍

  ```python
  .
  ├── offline_process
  │   ├── aos_schema.md                    # 创建aos chabot-index schema
  │   ├── aos_write_job.py                 # 离线数据注入脚本，通过S3 event触发glue执行
  │   ├── batch_upload_docs.py             # 批量数据注入脚本，手动执行
  │   └── chatbot_logs_func                # 对Cloudwatch输出的日志解析，通过KDF同步到OpenSearch (lambda 脚本)
  ```