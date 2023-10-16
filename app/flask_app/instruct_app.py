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

MAX_HISTORY = 5

def format_sse(data: str, event=None) -> str:
    msg = 'data: {}\n\n'.format(data)
    if event is not None:
        msg = 'event: {}\n{}'.format(event, msg)
    return msg

def get_model_location(model_name, deploy_region):

    base_path = os.path.dirname(os.path.realpath(__file__))
    model_location_dict = json.load(open(os.path.join(base_path, "model_location.json"), "r"))
    
    model_location = model_location_dict[deploy_region][model_name]
    return model_location
    

def start_server(bot, http_address: str, port: int, gpu_id: str):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    
    app = Flask(__name__)
    cors = CORS(app, supports_credentials=True)
    
    @app.route("/ping")
    def index():
        return Response(json.dumps({'message': 'started', 'success': True}, ensure_ascii=False), content_type="application/json")

    @app.route("/chat", methods=["GET", "POST"])
    def answer_question():
        result = {"query": "", "response": "", "success": False}
        try:
            if "application/json" in request.content_type:
                arg_dict = request.get_json()
                text = arg_dict["inputs"]
                params = arg_dict["parameters"]
                ori_history = arg_dict["history"]
                logger.info("Query - {}".format(text))
                if len(ori_history) > 0:
                    logger.info("History - {}".format(ori_history))
                history = ori_history[-MAX_HISTORY:]
                history = [tuple(h) for h in history]
                logger.info("New History - {}".format(history))
                response, history = bot.answer(text, history, params)
                logger.info("Answer - {}".format(response))
                ori_history.append((text, response))
                result = {"query": text, "outputs": response,
                          "history": ori_history, "success": True}
        except Exception as e:
            logger.error(f"error: {e}")
        return Response(json.dumps(result, ensure_ascii=False), content_type="application/json")

    @app.route("/stream", methods=["POST"])
    def answer_question_stream():
        def decorate(generator):
            for item in generator:
                yield format_sse(json.dumps(item, ensure_ascii=False), 'delta')
        result = {"query": "", "response": "", "success": False}
        text, history = None, None
        try:
            if "application/json" in request.content_type:
                arg_dict = request.get_json()
                text = arg_dict["inputs"]
                params = arg_dict["parameters"]
                ori_history = arg_dict["history"]
                logger.info("Query - {}".format(text))
                if len(ori_history) > 0:
                    logger.info("History - {}".format(ori_history))
                history = ori_history[-MAX_HISTORY:]
                history = [tuple(h) for h in history]
        except Exception as e:
            logger.error(f"error: {e}")
        return Response(decorate(bot.stream(text, history, params)), mimetype='text/event-stream')

    @app.route("/clear", methods=["GET", "POST"])
    def clear():
        history = []
        try:
            bot.clear()
            return Response(json.dumps({"success": True}, ensure_ascii=False), content_type="application/json")
        except Exception as e:
            return Response(json.dumps({"success": False}, ensure_ascii=False), content_type="application/json")

    logger.info("starting server...")
    server = pywsgi.WSGIServer((http_address, port), app)
    server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream API Service for InternLM')
    parser.add_argument('--device', '-d', help='device, -1 means cpu, other means gpu ids', default='0')
    parser.add_argument('--model', '-m', help='type of model, option: InternLM, Qwen', default=16)
    parser.add_argument('--host', '-H', help='host to listen', default='0.0.0.0')
    parser.add_argument('--port', '-P', help='port of this service', default=8800)
    parser.add_argument('--deploy_region', '-R', help='region of EC2', default='cn_deploy')
    args = parser.parse_args()

    model_location = get_model_location(args.model, args.deploy_region)

    if args.model == 'InternLM':
        from infer.instruct_internlm import InternLM
        bot = InternLM(gpu_id=args.device, model_location=model_location)
    elif args.model == 'Qwen':
        from infer.qwen_14b_int4 import Qwen14BInt4
        bot = Qwen14BInt4(gpu_id=args.device, model_location=model_location)

    start_server(bot, args.host, int(args.port), args.device)