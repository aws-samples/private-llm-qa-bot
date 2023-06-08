### LLM-QA-Knowledge-Split

- 运行依赖
    ```shell
    pip install boto3
    pip install BeautifulSoup4
    pip install langchain==0.0.162
    pip install pdfminer.six
    ```

- Knowledge Document(PDF) For Test 
  + https://docs.aws.amazon.com/zh_cn/AWSEC2/latest/UserGuide/ec2-ug.pdf
  + https://docs.aws.amazon.com/zh_cn/redshift/latest/dg/redshift-dg.pdf
  + https://docs.aws.amazon.com/zh_cn/whitepapers/latest/aws-overview/aws-overview.pdf

- Usage
    ```shell
    mkdir kg_dir
    # 指定 endpoint, 可以激活对长文本的摘要能力
    python doc_spliter.py --input_file ec2-ug.pdf --output_dir ./kg_dir --sep "=====" --title_level 4 --llm_endpoint ${llm_endpoint}
    ```