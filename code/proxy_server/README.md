## how to use
### start command
```bash
python3 -m price_server.py --host 0.0.0.0 --port 8001 --api-keys key1,key2
```

### how to run in backgroud

- 下载pm2  
sudo apt install npm
sudo npm install pm2 -g
pm2 start price_server.py --name price-app --interpreter python3 -- --host 0.0.0.0 --port 8001 --api-keys key1,key2
pm2 startup systemd
