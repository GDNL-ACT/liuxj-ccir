import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from collections import defaultdict
from vllm import LLM, SamplingParams

class PromptBuilder:
    def __init__(
            self, 
            system_prompt=(
                "你是一位精通中国法律体系的法律专家，专职为用户提供准确、专业且具有权威性的法律解答。"
                "你的任务是根据用户提出的问题，结合下方提供的法条材料，生成简明、直接且法律逻辑清晰的回答。\n\n"
                "请务必遵循以下规范：\n"
                "1. **精准引用法条**：优先从下方参考法条中选取最贴切的一条或几条，作为法律依据进行引用。引用时请注明法条名称与条号，例如：“根据《民法典》第xxx条规定”；\n"
                "2. **使用法律术语**：请使用通用、规范的法律术语和表达方式，避免使用口语化、模糊或日常化语言（如“应该吧”“大概可能”“常理上”）；\n"
                "3. **保持书面风格一致性**：如前文已有回答，请延续其正式、书面表达风格；如为首次回答，请保持结构清晰、逻辑严谨的正式法律说明文风；\n"
                "4. **精炼高效**：避免冗长、重复或泛泛而谈，直接切入法律核心内容，确保回答直击要点；\n"
                "5. **不得捏造法律内容**：若参考法条中无直接依据，请明确指出，并可根据一般法律原则谨慎说明，勿虚构或杜撰条款。\n\n"
                "以下是你可以参考的法条：\n"
            ),
            mode = 'default'
        ):
        self.mode = mode
        self.system_prompt = system_prompt

    def build_messages(self, history, current_question, articles):
        """
        构建 messages: [{"role": ..., "content": ...}]
        支持多轮问答历史、多条法条、角色设定
        """
        messages = []
        if self.system_prompt:
            if articles:
                law_context = "\n".join([
                    f"【法条{i+1}】{art['name']}：{art['content']}" for i, art in enumerate(articles)
                ])
                system_content = self.system_prompt  + law_context
            else:
                system_content = self.system_prompt
            messages.append({"role": "system", "content": system_content})

        if self.mode == 'rewrite':
            history_text = "\n".join(
                [f"用户：{h['question']}\n助手：{h['response']}" for h in history]
            )
            messages.append({
                "role": "user",
                "content": f"以下是之前的对话历史：\n{history_text}"
            })
        else:
            for i, turn in enumerate(history):
                question = turn.get("question", "")
                response = turn.get("response", "")
                if question:
                    messages.append({"role": "user", "content": f"{question}"})
                if response:
                    messages.append({"role": "assistant", "content": f"{response}"})

        if self.mode == 'rewrite':
            messages.append({"role": "user", "content": f"当前用户问题：{current_question}\n请你根据以上规则和历史对话，将用户的**最后一个问题**改写为一个脱离上下文也能独立理解的用于法律检索查询的问题"})
        else:
            messages.append({"role": "user", "content": current_question})
        return messages

class GeneratorVLLM:
    def __init__(self,
                 model_path: str,
                 max_articles: int = 5,
                 max_history : int = 4,
                 tensor_parallel_size : int = 1):
        self.prompt_builder = PromptBuilder()
        self.max_articles = max_articles
        self.max_history = max_history

        self.llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side='left',  
            trust_remote_code=True
        )
    
    def _generate(self, messages_batch, max_new_tokens=1024):
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
            # temperature=0.0,       
            # top_p=1.0,    
            temperature=0.6,
            top_k=20,
            top_p=0.95,
            max_tokens=max_new_tokens,
        )

        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)

        return [output.outputs[0].text.strip() for output in outputs]
    
    def run(self, input_path: str, output_path: str, batch_size: int):
        with open(input_path, 'r', encoding='utf-8') as f:
            dialog_data = json.load(f)

        turn_groups = defaultdict(list)
        for dialog in dialog_data:
            for turn_index, turn in enumerate(dialog["conversation"]):
                turn_groups[turn_index].append({
                    "dialog_id": dialog["id"],
                    "turn": turn_index,
                    "question": turn["question"],
                    # "recall": [r["article"] for r in turn.get("recall", [])],
                    "recall": [
                        {**r["article"], "score": r["score"]}
                        for r in turn.get("recall", [])
                    ], # 把score和article合并一下
                })

        histories = defaultdict(list)

        total_turns = sum(len(group) for group in turn_groups.values())
        pbar = tqdm(total=total_turns, desc="Generating responses")

        for turn_idx in sorted(turn_groups.keys()):
            items = turn_groups[turn_idx]
            messages_batch = []
            meta_batch = []

            for item in items:
                dialog_id = item["dialog_id"]
                question = item["question"]
                if self.max_articles == -1:
                    min_keep = 3
                    threshold = 0.55
                    # Step 1: 保留所有得分大于 threshold 的
                    recall_list = item["recall"]
                    filtered = [law for law in recall_list if law["score"] > threshold]
                    # Step 2: 如果数量不足 min_keep，则从 recall_list 头部补足（按 score 排序过的）
                    if len(filtered) < min_keep:
                        # 去重：防止添加重复条目
                        already_ids = {law["id"] for law in filtered}
                        for law in recall_list:
                            if law["id"] not in already_ids:
                                filtered.append(law)
                                already_ids.add(law["id"])
                            if len(filtered) >= min_keep:
                                break
                    recall = filtered
                else:
                    recall = item["recall"][:self.max_articles]
                history = histories[dialog_id][-self.max_history:]

                # 构建 prompt
                messages = self.prompt_builder.build_messages(history, question, recall)
                messages_batch.append(messages)
                meta_batch.append(item)

                if len(messages_batch) == batch_size:
                    responses = self._generate(messages_batch)
                    for meta, response in zip(meta_batch, responses):
                        meta["response"] = response
                        histories[meta["dialog_id"]].append({
                            "question": meta["question"],
                            "response": response
                        })
                    pbar.update(len(messages_batch))
                    messages_batch = []
                    meta_batch = []

            if messages_batch:
                responses = self._generate(messages_batch)
                for meta, response in zip(meta_batch, responses):
                    meta["response"] = response
                    histories[meta["dialog_id"]].append({
                        "question": meta["question"],
                        "response": response
                    })
                pbar.update(len(messages_batch))

        pbar.close()

        dialog_dict = {dialog["id"]: dialog for dialog in dialog_data}
        for dialog in dialog_data:
            dialog_id = dialog["id"]
            for turn_index, turn in enumerate(dialog["conversation"]):
                generated = histories[dialog_id][turn_index]["response"]
                turn["response"] = generated

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dialog_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # generator = GeneratorVLLM(
    #     model_path="/data/ccir-qwen3-32B"
    # )

    # generator.run(
    #     input_path="output/retrieval.json",
    #     output_path="output/gtmp.json",
    #     batch_size=128
    # )
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_articles", type=int, default=5)
    parser.add_argument("--max_history", type=int, default=4)
    args = parser.parse_args()
    
    generator = GeneratorVLLM(
        model_path=args.model_path,
        max_articles=args.max_articles,
        max_history=args.max_history,
        tensor_parallel_size=args.tensor_parallel_size
    )
    generator.run(
        input_path=args.input_path,
        output_path=args.output_path,
        batch_size=args.batch_size
    )
