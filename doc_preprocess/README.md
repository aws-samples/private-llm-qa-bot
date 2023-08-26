### OpenSource doc spliter

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
    # 指定 endpoint, 可以激活对长文本的摘要能力 --llm_endpoint ${llm_endpoint}
    python doc_spliter.py --input_file ec2-ug.pdf --output_dir ./kg_dir
    ```

### Textract-based doc spliter
    
- 运行依赖
    ```shell
    pip install boto3
    pip install BeautifulSoup4
    pip install pdfminer.six
    conda install -c conda-forge poppler
    pip install pdf2image
    pip install tqdm
    ```
    如果存在依赖问题， 参考https://github.com/Belval/pdf2image 的README中的安装步骤

- Knowledge Document(PDF) For Test 
  + ./common-stock-fs.pdf

- Usage
    ```shell
    # 指定 endpoint, 可以激活对长文本的摘要能力 --llm_endpoint ${llm_endpoint}
    python doc_spliter.py --input_file ../docs/common-stock-fs.pdf --output_file ../docs/common-stock-fs.pdf.json
    ```