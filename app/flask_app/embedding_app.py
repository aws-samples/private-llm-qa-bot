#! /usr/bin/python3
# -*- coding: utf-8 -*-

from gevent import monkey, pywsgi
monkey.patch_all()
from flask import Flask, request, Response
from flask_cors import CORS
import argparse
import logging
import os
import sys
import json

from infer.embedding_model import BGEEmbedding
from infer.cross_model import BufferCross

def getLogger(name, file_name, use_formatter=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s    %(message)s')
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    if file_name:
        handler = logging.FileHandler(file_name, encoding='utf8')
        handler.setLevel(logging.INFO)
        if use_formatter:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
            handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = getLogger('InternLM', 'chatlog.log')


def start_server(http_address: str, port: int, gpu_id: str):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    embedding_model = BGEEmbedding(gpu_id=gpu_id)
    cross_model = BufferCross(gpu_id=gpu_id)
    
    app = Flask(__name__)
    cors = CORS(app, supports_credentials=True)
    
    @app.route("/")
    def index():
        return Response(json.dumps({'message': 'started', 'success': True}, ensure_ascii=False), content_type="application/json")

    @app.route("/embedding", methods=["POST"])
    def generate_embedding():
        result = {"query": "", "response": "", "success": False}
        try:
            if "application/json" in request.content_type:
                arg_dict = request.get_json()
                input_sentences = None
                inputs = arg_dict["inputs"]
                if isinstance(inputs, list):
                    input_sentences = inputs
                else:
                    input_sentences =  [inputs]
                logging.info(f"inputs: {input_sentences}")

                result =  embedding_model.infer(input_sentences)
        except Exception as e:
            logger.error(f"error: {e}")
        return Response(json.dumps(result, ensure_ascii=False), content_type="application/json")
    
    @app.route("/cross", methods=["POST"])
    def generate_cross():
        result = {"query": "", "response": "", "success": False}
        try:
            if "application/json" in request.content_type:
                arg_dict = request.get_json()
                queries = arg_dict["inputs"]
                docs = arg_dict["docs"]
                logging.info(f"queries: {queries}, docs: {docs}")

                result =  cross_model.infer(queries, docs)
        except Exception as e:
            logger.error(f"error: {e}")
        return Response(json.dumps(result, ensure_ascii=False), content_type="application/json")

    @app.route("/clear", methods=["GET", "POST"])
    def clear():
        history = []
        try:
            embedding_model.clear()
            return Response(json.dumps({"success": True}, ensure_ascii=False), content_type="application/json")
        except Exception as e:
            return Response(json.dumps({"success": False}, ensure_ascii=False), content_type="application/json")

    @app.route("/score", methods=["GET"])
    def score_answer():
        score = request.get("score")
        logger.info("score: {}".format(score))
        return {'success': True}

    logger.info("starting server...")
    server = pywsgi.WSGIServer((http_address, port), app)
    server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream API Service for InternLM')
    parser.add_argument('--device', '-d', help='device，-1 means cpu, other means gpu ids', default='0')
    parser.add_argument('--host', '-H', help='host to listen', default='0.0.0.0')
    parser.add_argument('--port', '-P', help='port of this service', default=8800)
    args = parser.parse_args()
    start_server(args.host, int(args.port), args.device)