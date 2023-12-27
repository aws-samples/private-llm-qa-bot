FROM --platform=linux/amd64 python:3.9
WORKDIR /app
COPY ./requirements.txt .
RUN  pip3 install -r requirements.txt

COPY price_server.py /app
COPY get_price.py /app
COPY entrypoint.sh /app

RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]