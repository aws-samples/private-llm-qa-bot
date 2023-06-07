### LLM-QA-Knowledge-Split

- Knowledge Document(PDF) For Test 
  + https://docs.aws.amazon.com/zh_cn/AWSEC2/latest/UserGuide/ec2-ug.pdf
  + https://docs.aws.amazon.com/zh_cn/redshift/latest/dg/redshift-dg.pdf

- Usage
    ```shell
    mkdir kg_dir
    python doc_spliter.py --input_file ec2-ug.pdf --output_dir ./kg_dir --sep "=====" --title_level 4
    ```