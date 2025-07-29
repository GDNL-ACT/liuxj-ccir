import os
import json
import logging
import torch
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer
from generator_vllm import PromptBuilder
from tqdm import tqdm
from vllm import LLM, SamplingParams

class RewriterVLLM:
    def __init__(self, model_path: str, batch_size: int = 4, max_history : int = 4, tensor_parallel_size : int = 1):
        self.batch_size = batch_size
        self.max_history = max_history
        self.rewrite_builder = PromptBuilder(
            system_prompt=(
                "你是一个法律检索系统中的问题重写助手。"
                "请根据以下多轮对话内容（包括历史和当前问题），将用户的**最后一个问题**改写为一个脱离上下文也能独立理解的用于法律检索的查询。"
                "改写要求如下：\n"
                "1. **保留原问题的法律核心语义**，可以在不改变原意的前提下，适当使用常见或通用的法律术语，以增强表达的专业性；"
                "2. 不得引入对话中未出现的具体法律概念、事实推测或虚构信息；\n"
                "3. **清除所有指代词或模糊表述**（如“这个问题”“上述”“他”等），并结合历史对话**补全必要的背景信息**；\n"
                "4. 若原问题信息不足，只能从历史中提取已有事实进行补全，不得猜测或编造任何未提及信息;\n"
                "5. 改写后的问题必须是**完整、清晰、正式的陈述式问句**，具备良好的独立可读性和法律检索价值；\n"
                "6. **仅输出改写后的问题文本**，不附加任何解释、说明或前后缀内容;"
                "7.**改写后的问题应尽量简洁、清晰**，尽量在200字以内。"
            ),
            mode='rewrite'
        )
        self.llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side='left',
            trust_remote_code=True
        )
    
    def run(self, original_data_path: str, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(original_data_path, 'r', encoding='utf-8') as f:
            raw_list = [json.loads(line) for line in f]
        conv_map = defaultdict(list)
        for entry in raw_list:
            conv_map[entry.get("id")].append(entry)

        grouped = []
        for conv_id, entries in conv_map.items():
            sorted_entries = sorted(entries, key=lambda x: int(x.get("turn", 0)))
            grouped.append({
                "id": conv_id,
                "conversation": sorted_entries
            })
            
        processed = self._rewrite_question(grouped)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)
        logging.info(f"Process completed: {output_path}")

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
        
        sampling_params = SamplingParams(
            temperature=0.0,       
            top_p=1.0,            
            max_tokens=max_new_tokens
        )

        outputs = self.llm.generate(prompts, sampling_params,use_tqdm=False)

        return [output.outputs[0].text.strip() for output in outputs]
        
    def _rewrite_question(self, data_list):
        messages_list = []
        message_to_turn_ref = []

        for data in data_list:
            for turn in data.get("conversation", []):
                user_question = turn.get("user", "")
                history = [
                    {"question": h["user"], "response": h["assistant"]}
                    for h in turn.get("history", [])
                ]
                history = history[-self.max_history:]

                turn_index = data["conversation"].index(turn)
                rewrite_messages = self.rewrite_builder.build_messages(
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
                    "content": f"问题：{rewritten} \n答案：{data_ref['conversation'][turn_idx]['pseudo_answer']}"
                }
                
        return data_list
    

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
    
    processor = RewriterVLLM(
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_history=args.max_history,
        tensor_parallel_size=args.tensor_parallel_size
    )
    processor.run(
        original_data_path=args.data_path,
        output_path=args.output_path
    )