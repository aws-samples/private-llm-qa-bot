# Deploy LLM Models on EC2

## 描述：

技术栈选择
在设计推理搭建时，选择了Flask作为框架，构建推理的Web应用程序。为了满足国内环境对g5实例的不支持和显存限制的要求，当前的实现采用了两个g4dn实例上部署两个独立的Flask App。

Flask App 1：LLM推理/流式推理接口
该Flask App专门用于提供LLM（Large Language Model）推理和流式推理的接口。

Flask App 2：Embedding模型和Cross模型推理接口
该Flask App用于提供Embedding模型和Cross模型的推理接口。



## 需要的资源：

EC2：

* LLM Instance:
    * Instance type: g4dn.4xlarge
    * AMI: Deep Learning AMI GPU PyTorch 2.0.1 (Ubuntu 20.04) 20231003
    * Storage: 200GB
* Embedding & Cross Instance:
    * Instance type: g4dn.2xlarge
    * AMI: Deep Learning AMI GPU PyTorch 2.0.1 (Ubuntu 20.04) 20231003
    * Storage: 100GB



## 代码结构：

```
flask_app
├── embedding_app.py
├── instruct_app.py
├── infer
│   ├── __init__.py
│   ├── cross_model.py
│   ├── embedding_model.py
│   ├── instruct_model.py
│   └── requirements.txt
├── models
│   ├── cross
│   │   └── prepare_model.sh
│   ├── embedding
│   │   └── prepare_model.sh
│   └── instruct
│       └── prepare_model.sh
└── start_app.sh
```

* embedding_app.py 与instruct_app.py分别对应Embedding模型与LLM推理提供的接口
* infer路径下是Cross/Embedding/LLM的推理代码
* models路径下是Cross/Embedding/LLM的下载脚本
* start_app.sh是App的启动脚本，包括调用下载模型，安装依赖，启动App等等



## 部署步骤:

### 1. Start EC2 instances and Configure EC2

#### 方式1: 通过控制台创建EC2 （推荐）

* 打开 Amazon EC2 控制台。
* 选择 "Launch Instance（启动实例）"。
* 命名实例，例如"LLMInferenceEC2"。
* 在 "Amazon Machine Image (AMI)" 中，选择 Deep Learning AMI GPU PyTorch 2.0.1 (Ubuntu 20.04) 20231003。
* 在 "Instance Type" 中，选择 g4dn.4xlarge。
* 在 "Key pair (login)" 中，选择计划使用的密钥对。
* 在 "Network Settings" 中，选择计划使用的VPC与安全组。注意在这一步，需要确保安全组中开放了对应的端口。
* 在 "Configure Storage" 中，选择对应容量的存储。
* [Optional]如果客户需要用nginx进行其他配置，可以在 "Advanced Details" 中，在User Data部分完成。
```
    `#!/bin/bash
    apt update -y
    apt install nginx -y
    # set up nginx reverse proxy
    cd /etc/nginx/sites-available
    cat <<EOF > default
    server {
      listen 3000;
      server_name localhost;
      
      location /infer {
        proxy_pass http://127.0.0.1:3000;
      }
    }
    EOF
    systemctl enable nginx
    systemctl start nginx
    echo 'initialization completed' >> /home/ubuntu/userdata.log
```
* 选择 "Launch Instance"。

#### 方式2: 通过脚本创建EC2

```
aws ec2 run-instances --image-id $ami-id --count 2
--instance-type g4dn.4xlarge --key-name $key-pair 
--security-group-ids $security-group-id
--user-data file://userdata.sh 
--tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=$EC2Name}]' 
--query 'Instances[*].{ID:InstanceId,IP:PublicIpAddress}' 
--output text

```

### 2. 登入EC2，初始化conda环境&上传文件

```
# 本地上传文件到EC2
scp -i ec2_key.pem -r ./flask_app ubuntu@ec2_ip:/home/ubuntu/

# 登入EC2，并初始化conda环境
conda init
source ~/.bashrc

# 下载pm2
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash
. ~/.nvm/nvm.sh
nvm install --lts
npm install pm2 -g

```

### 3. 一键下载模型并启动Flask App

需要提供两个参数
    -t    对应的huggingface token
    -m    安装LLM App/Embedding App

例：
```
# global region
bash start_app.sh -m Qwen/BGE

# china region
bash start_app_cn.sh -m Qwen/BGE
```

```


### 

