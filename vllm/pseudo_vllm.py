import json
from typing import List, Dict
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from generator_vllm import PromptBuilder

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class PseudoAnswerGeneratorVLLM:
    def __init__(self, model_path, tensor_parallel_size : int = 1, max_history : int = 4):
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
        self.llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side='left',
            trust_remote_code=True
        )
        
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
                new_item = item.copy()
                new_item["history"] = history.copy()
                meta_batch.append(new_item)

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
        
        with open(output_path, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
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

        sampling_params = SamplingParams(
            temperature=0.0,       
            top_p=1.0,             
            max_tokens=max_new_tokens,
        )

        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)

        return [output.outputs[0].text.strip() for output in outputs]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_history", type=int, default=4)
    args = parser.parse_args()
    
    generator = PseudoAnswerGeneratorVLLM(
        model_path=args.model_path,
        max_history=args.max_history,
        tensor_parallel_size=args.tensor_parallel_size
    )
    generator.run(
        data_path=args.data_path,
        output_path=args.output_path,
        batch_size=args.batch_size
    )
