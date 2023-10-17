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

from infer.bge_zh import BGEEmbedding
from infer.buffer_cross import BufferCross

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

logger = getLogger('EmbeddingApp', 'chatlog.log')

def get_model_location(model_name, deploy_region):

    base_path = os.path.dirname(os.path.realpath(__file__))
    model_location_dict = json.load(open(os.path.join(base_path, "model_location.json"), "r"))
    
    model_location = model_location_dict[deploy_region][model_name]
    return model_location

def start_server(embedding_model, cross_model, http_address: str, port: int):
    
    app = Flask(__name__)
    cors = CORS(app, supports_credentials=True)
    
    @app.route("/ping")
    def index():
        return Response(json.dumps({'message': 'started', 'success': True}, ensure_ascii=False), content_type="application/json")

    @app.route("/embedding", methods=["POST"])
    def generate_embedding():
        result = {"sentence_embeddings": []}
        try:
            if "application/json" in request.content_type:
                arg_dict = request.get_json()
                input_sentences = None
                inputs = arg_dict["inputs"]
                is_query = arg_dict.get("is_query", False)
                instruction = arg_dict.get("instruction", "")
                if isinstance(inputs, list):
                    input_sentences = inputs
                else:
                    input_sentences =  [inputs]
                if is_query and instruction:
                    input_sentences = [ instruction + sent for sent in input_sentences ]
                logging.info(f"inputs: {input_sentences}")

                result =  embedding_model.infer(input_sentences)
        except Exception as e:
            logger.error(f"error: {e}")
        return Response(json.dumps(result, ensure_ascii=False), content_type="application/json")
    
    @app.route("/cross", methods=["POST"])
    def generate_cross():
        result = {"scores": [], "success": False}
        try:
            if "application/json" in request.content_type:
                if cross_model:
                    arg_dict = request.get_json()
                    queries = arg_dict["inputs"]
                    docs = arg_dict["docs"]
                    logging.info(f"queries: {queries}, docs: {docs}")

                    result =  cross_model.infer(queries, docs)
                else:
                    result = {"scores": [], "success": False, "message": "cross model is not enabled"}
        except Exception as e:
            logger.error(f"error: {e}")
            result = {"scores": [], "success": False, "message": str(e)}
        return Response(json.dumps(result, ensure_ascii=False), content_type="application/json")

    @app.route("/clear", methods=["GET", "POST"])
    def clear():
        history = []
        try:
            embedding_model.clear()
            return Response(json.dumps({"success": True}, ensure_ascii=False), content_type="application/json")
        except Exception as e:
            return Response(json.dumps({"success": False}, ensure_ascii=False), content_type="application/json")

    logger.info("starting server...")
    server = pywsgi.WSGIServer((http_address, port), app)
    server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream API Service for Embedding Model')
    parser.add_argument('--device', '-d', help='device，-1 means cpu, other means gpu ids', default='0')
    parser.add_argument('--host', '-H', help='host to listen', default='0.0.0.0')
    parser.add_argument('--port', '-P', help='port of this service', default=8800)
    parser.add_argument('--cross', '-C', help='whether to enable cross model', default=False)
    parser.add_argument('--deploy_region', '-R', help='region of EC2', default='cn_deploy')
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    embedding_model_location = get_model_location("BGE", args.deploy_region)
    cross_model_location = get_model_location("Cross", args.deploy_region)

    embedding_model = BGEEmbedding(gpu_id=args.device, model_location=embedding_model_location)
    cross_model = BufferCross(gpu_id=args.device, model_location=cross_model_location) if args.cross else None

    start_server(embedding_model, cross_model, args.host, int(args.port))