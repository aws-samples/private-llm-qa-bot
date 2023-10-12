import torch
import os
from sentence_transformers import SentenceTransformer

# os.environ["CUDA_VISIBLE_DEVICES"] = ""

model_name="models--BAAI--bge-large-zh-v1.5"
commit_hash="00f8ffc4928a685117583e2a38af8ebb65dcec2c"
base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
model_location = os.path.join(base_path, "models", "embedding", model_name, "snapshots", commit_hash)

class BGEEmbedding():
    def __init__(self, gpu_id) -> None:
        self.model = self._model(gpu_id)
        response = self.model.encode(["你好", "你是谁"], normalize_embeddings=True)
    
    def _model(self, gpu_id):
        model = None
        if gpu_id == '-1':
            device = torch.device('cpu')
            self.devices = []
            model = SentenceTransformer(model_location)
            model.to(device)
        else:
            gpu_ids = gpu_id.split(",")
            self.devices = ["cuda:{}".format(id) for id in gpu_ids]
            model = SentenceTransformer(model_location)
            model.eval().cuda()

        return model
    
    def clear(self) -> None:
        if torch.cuda.is_available():
            for device in self.devices:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
    
    def infer(self, input_sentences):

        sentence_embeddings =  self.model.encode(input_sentences, normalize_embeddings=True)
            
        result = {"sentence_embeddings": sentence_embeddings.tolist()}
        return result

if __name__ == "__main__":
    embedding_model = BGEEmbedding(gpu_id="0")
    result = embedding_model.infer(["你好", "你是谁"])
    print(result)