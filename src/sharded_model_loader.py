# model_loader.py

import os
import shutil
import torch
import requests
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open


class ModelLoader:
    def __init__(self, model_name, model_id, cache_dir):
        self.model_name = model_name
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_config(self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_shard_to_gpu(self, shard_file):
        with safe_open(shard_file, framework="pt") as f:
            for name in f.keys():
                tensor = f.get_tensor(name).to(self.device)
                if name in self.model.state_dict():
                    self.model.state_dict()[name].copy_(tensor)
                else:
                    print(f"Warning: Tensor {name} not found in model state_dict")

    def load_local_shards(self, shard_local_files):
        for shard_file in shard_local_files:
            print(f"Loading shard: {shard_file}")
            self.load_shard_to_gpu(shard_file)
        return self.model, self.tokenizer

    def download_shard(self, url, local_file, token):
        print(f"Downloading shard from {url}")
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(url, headers=headers, stream=True)
        with open(local_file, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        print(f"Shard downloaded to {local_file}")

    def download_and_load_shards(self, shard_urls, shard_local_files, token):
        for url, local_file in zip(shard_urls, shard_local_files):
            print(f"Processing shard: {url}")
            self.download_shard(url, local_file, token)
            self.load_shard_to_gpu(local_file)
            os.remove(local_file)
        return self.model, self.tokenizer
