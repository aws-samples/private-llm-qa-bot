#!/bin/bash
pm2 start price_server.py --name price-app --interpreter python3 -- --host 0.0.0.0 --port 8001 --api-keys apikey1
pm2 startup systemd