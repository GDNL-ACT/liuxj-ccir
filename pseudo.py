import torch
import json
from typing import List, Dict
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from tqdm import tqdm

class PseudoAnswerGenerator:
    def __init__(self, model_path , max_history : int = 4):
        self.prompt_builder = PromptBuilder(
            system_prompt=(
                "你是一位精通中国法律体系的法律专家，专职为用户提供准确、专业且具有权威性的法律解答。"
                "你的任务是根据用户提出的问题，生成简明、直接且法律逻辑清晰的回答。\n\n"
                "请务必遵循以下规范：\n"
                "1. **使用法律术语**：请使用通用、规范的法律术语和表达方式，避免使用口语化、模糊或日常化语言（如“应该吧”“大概可能”“常理上”）；\n"
                "2. **精炼高效**：避免冗长、重复或泛泛而谈，直接切入法律核心内容，确保回答直击要点；\n"
                "3. 仅输出本轮问题(即最后一个)的回答内容，不输出任何多余说明或注释。回答尽量在500字以内\n"
            )
        )
        self.max_history = max_history
        self.model_path = model_path
        self._ensure_model_loaded()
        
    def run(self, data_path: str,  output_path: str, batch_size: int = 4,):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        turn_groups = defaultdict(list)
        for item in data:
            turn_groups[int(item["turn"])].append(item)

        histories = defaultdict(list)
        results = []

        total_items = sum(len(items) for items in turn_groups.values())
        pbar = tqdm(total=total_items, desc="Pseudo answer generating")

        for turn in sorted(turn_groups.keys()):
            items = turn_groups[turn]
            messages_batch = []
            meta_batch = []

            for item in items:
                dialog_id = item["id"]
                user_question = item["user"]
                history = histories[dialog_id][-self.max_history:]
                pseudo_history = [
                    {"question": h["user"], "response": h["assistant"]}
                    for h in history
                ]
                messages = self.prompt_builder.build_messages(pseudo_history, user_question, articles=[])
                messages_batch.append(messages)
                meta_batch.append({
                    "id": dialog_id,
                    "turn": str(turn),
                    "user": user_question,
                    "history": history.copy()
                })

                if len(messages_batch) == batch_size:
                    outputs = self._generate_with_prompt_builder(messages_batch)
                    for m, output in zip(meta_batch, outputs):
                        m["pseudo_answer"] = output
                        histories[m["id"]].append({"user": m["user"], "assistant": output})
                        results.append(m)
                    pbar.update(len(messages_batch))
                    messages_batch = []
                    meta_batch = []

            if messages_batch:
                outputs = self._generate_with_prompt_builder(messages_batch)
                for m, output in zip(meta_batch, outputs):
                    m["pseudo_answer"] = output
                    histories[m["id"]].append({"user": m["user"], "assistant": output})
                    results.append(m)
                pbar.update(len(messages_batch))
        pbar.close()
        
        if output_path:
            self._save_to_file(results, output_path)
    
    def _generate_with_prompt_builder(self, messages_batch, max_new_tokens=512):
        prompts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            for messages in messages_batch
        ]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
            )

        decoded_outputs = []
        for i, output in enumerate(outputs):
            decoded = self.tokenizer.decode(output[len(inputs["input_ids"][i]):], skip_special_tokens=True)
            decoded_outputs.append(decoded.strip())

        return decoded_outputs

    def _save_to_file(self, results: List[Dict], file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _ensure_model_loaded(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side='left',  
            trust_remote_code=True 
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",              
            device_map="auto",                   
            trust_remote_code=True
        )
        self.model.eval()

if __name__ == "__main__":
    from generator import PromptBuilder
    generator = PseudoAnswerGenerator(model_path="/home/liuxj25/LawLLM/CCIR/models/Qwen3-32B")
    generator.run(
        data_path="/home/liuxj25/LawLLM/CCIR/data/question_A.json", 
        output_path="output/pseudo_answers.jsonl",
        batch_size = 16
    )
