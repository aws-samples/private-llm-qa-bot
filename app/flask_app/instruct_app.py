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

from infer.instruct_model import InternLM

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


def start_server(quantize_level, http_address: str, port: int, gpu_id: str):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    bot = InternLM(gpu_id=gpu_id)
    
    app = Flask(__name__)
    cors = CORS(app, supports_credentials=True)
    
    @app.route("/")
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
                result = {"query": text, "response": response,
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
    parser.add_argument('--quantize', '-q', help='level of quantize, option：16, 8 or 4', default=16)
    parser.add_argument('--host', '-H', help='host to listen', default='0.0.0.0')
    parser.add_argument('--port', '-P', help='port of this service', default=8800)
    args = parser.parse_args()
    start_server(args.quantize, args.host, int(args.port), args.device)