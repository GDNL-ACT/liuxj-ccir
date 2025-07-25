import json

import json

BASE_SYSTEM_PROMPT = (
    "你是一位精通中国法律体系的法律专家，专职为用户提供准确、专业且具有权威性的法律解答。"
    "你的任务是根据用户提出的问题，结合下方提供的法条材料，生成简明、直接且法律逻辑清晰的回答。"
)

# github 数据集
def convert_format_dataset0(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    output_data = []

    for item in raw_data:
        user_q = item["user"]
        assistant_a = item["assistant"]

        system_prompt = BASE_SYSTEM_PROMPT

        article_context = item.get("article_context", [])
        if article_context:
            law_contexts = "\n".join([
                f"【法条{i+1}】{title}：{content}"
                for i, art in enumerate(article_context)
                for title, content in art.items()
                if content is not None
            ])
            system_prompt += "\n以下是你可以参考的法条：\n" + law_contexts

        sample = {
            "instruction": user_q,
            "input": "",
            "output": assistant_a,
            "system": system_prompt,
            "history": []
        }
        output_data.append(sample)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    return output_data

# dataset1 
def convert_format_dataset1(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    output_data = []

    for item in raw_data:
        turns = item["conversation"]
        history = []

        for idx, turn in enumerate(turns):
            system_prompt = BASE_SYSTEM_PROMPT
            user_q = turn["user"]
            assistant_a = turn["assistant"]

            article_context = turn.get("article_context", [])
            if article_context:
                law_contexts = "\n".join([
                    f"【法条{i+1}】{title}：{content}"
                    for i, art in enumerate(article_context)
                    for title, content in art.items()
                    if content is not None
                ])
                system_prompt += "以下是你可以参考的法条：\n" + law_contexts

            sample = {
                "instruction": user_q,
                "input": "",
                "output": assistant_a,
                "system": system_prompt,
                "history": history.copy()
            }
            output_data.append(sample)

            history.append([user_q, assistant_a])

    with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    return output_data

# ppo dataset1
def convert_format_ppo_dataset1(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    output_data = []

    for item in raw_data:
        turns = item["conversation"]
        history = []

        for idx, turn in enumerate(turns):
            system_prompt = BASE_SYSTEM_PROMPT
            user_q = turn["user"]
            assistant_a = turn["assistant"]

            article_context = turn.get("article_context", [])
            if article_context:
                law_contexts = "\n".join([
                    f"【法条{i+1}】{title}：{content}"
                    for i, art in enumerate(article_context)
                    for title, content in art.items()
                    if content is not None
                ])
                system_prompt += "以下是你可以参考的法条：\n" + law_contexts

            sample = {
                "instruction": user_q,
                "input": "",
                "output": assistant_a,
                "system": system_prompt,
                "history": history.copy(),
                "keywords": turn["keyword"]
            }
            output_data.append(sample)

            history.append([user_q, assistant_a])

    with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    return output_data

if __name__ == "__main__":
    ## 示例用法
    # d1 = convert_format_dataset1("/home/liuxj25/LawLLM/CCIR/eval/output/dataset1_rewritten.json", "../data/ccir_generation_multiturn.json")
    # d0 = convert_format_dataset0("/home/liuxj25/LawLLM/CCIR/data/dataset0.json", "../data/ccir_generation_oneturn.json")
    # with open("../data/ccir_generation.json", "w", encoding="utf-8") as f:
    #         json.dump(d1+d0, f, ensure_ascii=False, indent=2)
    convert_format_ppo_dataset1("/home/liuxj25/LawLLM/CCIR/eval/output/dataset1_rewritten.json", "../data/ccir_generation_ppo.json")