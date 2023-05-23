# QA-chatbot-workshop部署文档  
## 系统架构  
![image](https://user-images.githubusercontent.com/19160090/237026376-bc9cb0ea-f4cd-4a12-a51d-0d7a99bb2eda.png)

## 步骤  
### 下载代码  
git clone 
### 安装依赖 
1. 安装nodejs 18.x  
2. 进入deploy目录，运行以下命令, 安装依赖的moudels 
```cd deploy  
sudo npm install -g aws-cdk  
npm install  
```
### 修改环境变量  
修改环境变量文件deploy/ .env.sample, 补充完信息之后, 将文件名修改成.env  
`mv .env.sample .env`  
注意：   
1. existing_vpc_id和aos_existing_endpoint 是可选，如果不提供，则创建新的vpc和aos   
2. llm_xxx_endpoint根据名称填写   
![image](https://user-images.githubusercontent.com/19160090/237028301-138d9acf-4744-47dc-a374-59609cef7b41.png)
填写实际例子：  
![image](https://user-images.githubusercontent.com/19160090/237028300-1300c4e9-30f8-432d-800b-89aca09b0481.png)

## 部署安装  
进入code目录，打包lambda image （由于安装了langchain的包，体积较大，无法通过zip上传）， 使用以下脚本打包成image并上传ECR  
安装docker(如果有docker 则可跳过，以下是ubuntu安装方式)  
```sudo snap install docker  
sudo chmod 666 /var/run/docker.sock  
```
```
cd code
bash build_and_push.sh {REGION} {PROFILE_NAME}
```

{PROFILE_NAME} 是AWS CLI profile，可选参数，如果不填默认使用default profile  

部署安装  
```
cdk bootstrap --profile {PROFILE_NAME}  
cdk synth --profile {PROFILE_NAME}  
cdk deploy --profile {PROFILE_NAME}  
```