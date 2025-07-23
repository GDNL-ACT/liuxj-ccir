import os
import json
import logging
import torch
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from generator import PromptBuilder
from tqdm import tqdm

class Processor:
    def __init__(self, process_type: str, batch_size: int = 4, model_path: str = None):
        self.process_type = process_type
        self.process_methods = {
            "current_question": self._current_question,
            "prefix_question": self._prefix_question,
            "rewrite_question": self._rewrite_question,
        }
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.batch_size = batch_size
        
    def _generate_with_prompt_builder(self, messages_batch, max_new_tokens=128):
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
            decoded_outputs.append(decoded)
        return decoded_outputs

    def _rewrite_question(self, data_list):
        self._ensure_model_loaded()
        rewrite_builder = PromptBuilder(
            system_prompt=(
                "你是一个法律检索系统中的问题重写助手。"
                "请根据以下多轮对话内容（包括历史对话和当前问题），将用户的最后一个问题改写为一个脱离上下文也能独立理解的法律检索查询。"
                "改写要求如下：\n"
                "1. **保留原问题的法律核心语义**，可以在不改变原意的前提下，适当使用常见或通用的法律术语，以增强表达的专业性；"
                "2. 不得引入对话中未出现的具体法律概念、事实推测或虚构信息；\n"
                "3. **清除所有指代词或模糊表述**（如“这个问题”“上述”“他”等），并结合历史对话**补全必要的背景信息**；\n"
                "4. 改写后的问题必须是**完整、清晰、正式的陈述式问句**，具备良好的独立可读性和法律检索价值；\n"
                "5. 若原问题信息不足，应从历史对话中提取明确信息补全，不得随意扩展；\n"
                "6. **仅输出改写后的问题文本**，不附加任何解释、说明或前后缀内容。**改写后的问题应尽量简洁、清晰**，尽量在250字以内。"
            )        
        )

        messages_list = []
        message_to_turn_ref = []

        for data in data_list:
            for turn in data.get("conversation", []):
                user_question = turn.get("user", "")
                history = [
                    {"question": h["user"], "response": h["assistant"]}
                    for h in turn.get("history", [])
                ]
                turn_index = data["conversation"].index(turn)

                rewrite_messages = rewrite_builder.build_messages(
                    history,
                    current_question=user_question,
                    articles=[]
                )
                messages_list.append(rewrite_messages)
                message_to_turn_ref.append((data, turn_index))

        for i in tqdm(range(0, len(messages_list), self.batch_size), desc="Rewriting queries"):
            batch_messages = messages_list[i:i + self.batch_size]
            rewritten_outputs = self._generate_with_prompt_builder(batch_messages)

            for j, rewritten in enumerate(rewritten_outputs):
                data_ref, turn_idx = message_to_turn_ref[i + j]
                data_ref["conversation"][turn_idx]["query"] = {
                    "type": "rewrite_question",
                    "content": rewritten.strip()
                }
     
        return data_list
    
    def run(self, original_data_path: str, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if self.process_type != "rewrite_question":
            with open(original_data_path, "r", encoding="utf-8") as f:
                raw_list = json.load(f)
        else:
            with open(original_data_path, 'r', encoding='utf-8') as f:
                raw_list = [json.loads(line) for line in f]
        conv_map = defaultdict(list)
        for entry in raw_list:
            conv_map[entry.get("id")].append(entry)

        grouped = []
        for conv_id, entries in conv_map.items():
            try:
                sorted_entries = sorted(entries, key=lambda x: int(x.get("turn", 0)))
            except Exception:
                sorted_entries = entries
            grouped.append({
                "id": conv_id,
                "conversation": sorted_entries
            })
            
        method = self.process_methods.get(self.process_type)
        processed = method(grouped)

        output_list = []
        for conv in processed:
            conv_id = conv.get("id")
            conversation = [
                {
                    "user": turn.get("user", ""),
                    "query": turn.get("query", {}),
                }
                for turn in conv.get("conversation", [])
            ]
            output_list.append({
                "id": conv_id,
                "conversation": conversation
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_list, f, ensure_ascii=False, indent=2)
        logging.info(f"Process completed: {output_path}")

    def _current_question(self, data_list):
        for data in data_list:
            for conversation in data.get("conversation", []):
                conversation["query"] = {
                    "type": "current_question",
                    "content": conversation.get("user", ""),
                }
        return data_list

    def _prefix_question(self, data_list):
        for data in data_list:
            prefix = ""
            for conversation in data.get("conversation", []):
                prefix += conversation.get("user", "")
                conversation["query"] = {
                    "type": "prefix_question",
                    "content": prefix,
                }
        return data_list
    
    def _ensure_model_loaded(self):
        if self.model is None or self.tokenizer is None:
            logging.info(f"Loading model from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
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
    processor = Processor("rewrite_question", model_path="/home/liuxj25/LawLLM/CCIR/models/Qwen3-4B")
    processor.run("output/pseudo_answers.json", "output/rewritten_queries.json")

    # processor = Processor("prefix_question")
    # processor.run("../data/qut.json", "output/tmp.jsonl")
