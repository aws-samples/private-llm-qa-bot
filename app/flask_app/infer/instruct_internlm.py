import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# model_name="models--csdc-atl--buffer-instruct-InternLM-001"
# commit_hash="2da398b96f1617c22af037e9177940cc1c823fcf"
# base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# model_location = os.path.join(base_path, "models", "instruct", model_name, "snapshots", commit_hash)

class InternLM():
    def __init__(self, gpu_id, model_location) -> None:
        self.model_location = model_location
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_location, trust_remote_code=True)
        self.model = self._model(gpu_id)
        response, history = self.model.chat(self.tokenizer, "你好", history=[])
        print(response)
    
    def _model(self, gpu_id):
        model = None
        if gpu_id == '-1':
            # Inference on CPU is not supported.
            # Throw out an error message
            raise RuntimeError("Inference on CPU is not supported.")
        else:
            gpu_ids = gpu_id.split(",")
            self.devices = ["cuda:{}".format(id) for id in gpu_ids]
            model = AutoModelForCausalLM.from_pretrained(self.model_location, trust_remote_code=True).eval().half().cuda()

        return model
    
    def clear(self) -> None:
        if torch.cuda.is_available():
            for device in self.devices:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
    
    def answer(self, query: str, history, params):
        response, history = self.model.chat(self.tokenizer, query, history=history, **params)
        history = [list(h) for h in history]
        return response, history

    def stream(self, query, history, params):
        if query is None or history is None:
            yield {"query": "", "outputs": "", "response": "", "history": [], "finished": True}
        size = 0
        response = ""
        for response in self.model.stream_chat(self.tokenizer, query, history, **params):
            this_response = response[size:]
            size = len(response)
            yield {"outputs": this_response, "response": response, "finished": False}
        yield {"query": query, "outputs": "[EOS]", "response": response[-1][-1], "history": response, "finished": True}

if __name__ == "__main__":
    model_name="models--csdc-atl--buffer-instruct-InternLM-001"
    commit_hash="2da398b96f1617c22af037e9177940cc1c823fcf"
    base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    model_location = os.path.join(base_path, "models", "instruct", model_name, "snapshots", commit_hash)

    bot = InternLM(gpu_id="0", model_location=model_location)
    