## 临时说明，用于在中国区搭建一个查询价格的接口转发服务器
## how to use

### 中国区启动一台ec2 unbuntu instance
### start command
```bash
python3 price_server.py --host 0.0.0.0 --port 8001 --api-keys key1,key2
```
### how to run in backgroud

- 下载pm2
```bash
sudo apt install npm
sudo npm install pm2 -g

##修改launch.sh 里的key
vim launch.sh
##启动
bash launch.sh
```
- 


### run in ECS
- 1. 修改 [entrypoint.sh](./entrypoint.sh) 里的api key
- 2. bash [build_and_push.sh](./build_and_push.sh) cn-northwest-1
- 3. 使用 [cf_template.yaml](./cf_template.yaml) 创建资源, 需要制定vpc，sg，publicSubnetid
- 4. CF stack 里输出的ALB的地址 如 http://xxx.cn-northwest-1.elb.amazonaws.com.cn:8001/v1/get_ec2_price


