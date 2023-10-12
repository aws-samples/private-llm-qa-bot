import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig

model_name="models--Qwen--Qwen-14B-Chat-Int4"
commit_hash="0f5e18f5f18b3ced68be965099f091189964ed85"
base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
model_location = os.path.join(base_path, "models", "instruct", model_name, "snapshots", commit_hash)

class Qwen14BInt4():
    def __init__(self, gpu_id) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_location, trust_remote_code=True)
        self.model, self.config = self._model(gpu_id)
        response, history = self.model.chat(self.tokenizer, "你好", history=[])
        print(response)
    
    def _model(self, gpu_id):
        model = None
        config = None
        if gpu_id == '-1':
            # Inference on CPU is not supported.
            # Throw out an error message
            raise RuntimeError("Inference on CPU is not supported.")
        else:
            gpu_ids = gpu_id.split(",")
            self.devices = ["cuda:{}".format(id) for id in gpu_ids]
            model = AutoModelForCausalLM.from_pretrained(model_location, device_map="auto", trust_remote_code=True).eval()
            model.generation_config  = GenerationConfig.from_pretrained(model_location, trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参
    
            return model, model.generation_config 

        return model, config
    
    def clear(self) -> None:
        if torch.cuda.is_available():
            for device in self.devices:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
    
    def answer(self, query: str, history, params):
        self.config.max_new_tokens = params.get('max_length',1024)
        self.config.top_p = params.get('top_p',1)
        response, history = self.model.chat(self.tokenizer, query, history=history, generation_config=self.config)
        history = [list(h) for h in history]
        return response, history

    def stream(self, query, history, params):
        if query is None or history is None:
            yield {"query": "", "response": "", "history": [], "finished": True}
        size = 0
        self.config.max_new_tokens = params.get('max_length',1024)
        self.config.top_p = params.get('top_p',1)
        response = ""
        for response in self.model.chat_stream(self.tokenizer, query, history, generation_config=self.config):
            this_response = response[size:]
            size = len(response)
            yield {"delta": this_response, "response": response, "finished": False}
        history.append([query, response])
        yield {"query": query, "delta": "[EOS]", "response": response, "history": history, "finished": True}

if __name__ == "__main__":
    bot = Qwen14BInt4(gpu_id="0")
    result = bot.stream(query = "请问S3是怎么计费的？", history = [], params = {})
    