import torch
import logging
import math
import os
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_name="models--csdc-atl--buffer-cross-001"
commit_hash="46d270928463db49b317e5ea469a8ac8152f4a13"
base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
model_location = os.path.join(base_path, "models", "cross", model_name, "snapshots", commit_hash)

class BufferCross():
    def __init__(self, gpu_id) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_location, use_fast=False)
        self.model = self._model(gpu_id)
        response = self.infer("请问AWS Clean Rooms是多方都会收费吗？", "请问AWS Clean Rooms是多方都会收费吗？")
    
    def _model(self, gpu_id):
        model = None
        if gpu_id == '-1':
            self.device = torch.device('cpu')
            self.devices = []
            model = AutoModel.from_pretrained(model_location, trust_remote_code=True).half()
            model.to(device)
            model.requires_grad_(False)
            model.eval()
        else:
            gpu_ids = gpu_id.split(",")
            self.device = torch.device('cuda:{}'.format(gpu_ids[0]))
            self.devices = ["cuda:{}".format(id) for id in gpu_ids]
            model = AutoModel.from_pretrained(model_location, trust_remote_code=True).half()
            model.to(device)
            model.requires_grad_(False)
            model.eval()

        return model
    
    def clear(self) -> None:
        if torch.cuda.is_available():
            for device in self.devices:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
    
    def infer(self, queries, docs):

        encoded_input = self.tokenizer(text = [queries], text_pair=[docs], padding=True, truncation=True, max_length=2048, return_tensors='pt')['input_ids'].to(self.device)
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(input_ids=encoded_input)
        
        result = {"scores": model_output.cpu().numpy().tolist()}
        return result


if __name__ == "__main__":
    # test
    data = {
        "inputs": "请问AWS Clean Rooms是多方都会收费吗？",
        "docs": "请问AWS Clean Rooms多方都会收费吗？"
    }
    cross_model = BufferCross(gpu_id="0")
    result = cross_model.infer(data["inputs"], data["docs"])
    print(result)