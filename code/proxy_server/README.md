## 临时说明，用于在中国区搭建一个查询价格的接口转发服务器
## how to use

### 中国区启动一台ec2 unbuntu instance
### start command
```bash
python3 -m price_server.py --host 0.0.0.0 --port 8001 --api-keys key1,key2
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

