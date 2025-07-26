import json
import logging
import faiss
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import shutil
import os
from datetime import datetime
import torch
import torch.nn.functional as F

    
class Retriever:
    def __init__(self,
                 model_path: str = None,
                 batch_size: int = 16,
                 lora_path: str = None,
                 index_path:str = "output/law_index.faiss"):
        self.model_path = model_path
        self.lora_path = lora_path
        self.batch_size = batch_size
        self.index_path = index_path

        if lora_path is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True,
                padding_side="left"
            )
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype="auto",              
                device_map="auto", 
                trust_remote_code=True
            )
            self.model.eval()
        else:
            self.model, self.tokenizer = self._load_lora(model_path, lora_path)
    
    def _load_lora(self, base_path, lora_path, merged_dir_root: str = "models"):
        now_str = datetime.now().strftime("%Y%m%d_%H%M")
        merged_dir = os.path.join(merged_dir_root, now_str)
        os.makedirs(merged_dir, exist_ok=True)

        base_model = AutoModel.from_pretrained(
            base_path,
            torch_dtype="auto",              
            device_map="auto", 
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, lora_path)
        model = model.merge_and_unload()
        model.save_pretrained(merged_dir)

        tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True, padding_side="left")
        tokenizer.save_pretrained(merged_dir)
        model.eval()
        return model, tokenizer

    def _embed(self, texts: list, faiss_index=None) -> np.ndarray:
        def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
            
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding"):
            batch = texts[i: i + self.batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeds = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
                embeds = F.normalize(embeds, p=2, dim=1)
                embeds = embeds.to(torch.float32).cpu().numpy()
            if faiss_index is not None:
                faiss_index.add(embeds)
            else:
                embeddings.append(embeds)
        if faiss_index is None:
            return np.vstack(embeddings)
        else:
            return None

    def run(self,
            input_path: str,
            law_path: str,
            output_path: str,
            top_k: int = 10):
        with open(input_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)

        index = self._build_index(law_path)

        results = self._search(conversations, law_path, top_k=top_k, idx=index)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved retrieval results to {output_path}")

    def _build_index(self, law_path: str):
        with open(law_path, 'r', encoding='utf-8') as f:
            corpus = [json.loads(line)["name"]+" "+ json.loads(line)["content"] for line in f]
        
        dim = self.model.config.hidden_size
        idx = faiss.IndexFlatIP(dim)
        self._embed(corpus, faiss_index=idx)
        logging.info(f"Law library FAISS index bulided")
        return idx

    def _search(self,
               data: list,
               law_path: str,
               top_k: int = 5,
               idx = None) -> list:
        if idx is None:
            raise ValueError("Index must be passed.")
        qs = [conv["query"]["content"] for d in data for conv in d["conversation"]]
        q_embs = self._embed(qs)
        D, I = idx.search(q_embs.astype('float32'), top_k)

        with open(law_path, 'r', encoding='utf-8') as f:
            laws = [json.loads(line) for line in f]

        query_idx = 0
        for d in data:
            for conv in d["conversation"]:
                recalls = []
                for j in range(top_k):
                    recalls.append({
                        "article": laws[I[query_idx][j]],
                        "score": float(D[query_idx][j])
                    })
                conv["recall"] = recalls
                conv["response"] = ""
                if "query" in conv:
                    del conv["query"]
                if "pseudo_answer" in conv:
                    del conv["pseudo_answer"]
                if "user" in conv:
                    conv["question"] = conv.pop("user")
                query_idx += 1
        return data

if __name__ == "__main__":
    retriever = Retriever(
        model_path = "/home/liuxj25/LawLLM/CCIR/eval/models/20250726_0537",
        index_path="/home/liuxj25/LawLLM/CCIR/eval/output/law_index.faiss",
        batch_size=64
    )

    retriever.run(
        input_path="/home/liuxj25/LawLLM/CCIR/eval/output/queries.json",
        law_path="/home/liuxj25/LawLLM/CCIR/data/law_library.jsonl",
        output_path="/home/liuxj25/LawLLM/CCIR/eval/output/retrieval.json",
        top_k=5
    )

    


