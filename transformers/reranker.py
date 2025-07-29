import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

# === 参数配置 ===
model_name = "Qwen/Qwen3-Reranker-8B"
input_path = "input.jsonl"
output_path = "output.jsonl"
batch_size = 8
instruction = "Given a web search query, retrieve relevant passages that answer the query"
max_length = 8192

# === 初始化模型 ===
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_name).cuda().eval()

token_true_id = tokenizer.convert_tokens_to_ids("yes")
token_false_id = tokenizer.convert_tokens_to_ids("no")

prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

# === 构造 Prompt ===
def format_instruction(instruction, query, doc):
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

# === Tokenizer 输入处理 ===
def process_inputs(pairs):
    inputs = tokenizer(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
    )
    for i in range(len(inputs["input_ids"])):
        inputs["input_ids"][i] = prefix_tokens + inputs["input_ids"][i] + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    return {k: v.cuda() for k, v in inputs.items()}

# === Logits 得分计算 ===
@torch.no_grad()
def compute_logits(inputs):
    logits = model(**inputs).logits[:, -1, :]
    score_yes = logits[:, token_true_id]
    score_no = logits[:, token_false_id]
    scores = torch.stack([score_no, score_yes], dim=1)
    probs = F.softmax(scores, dim=1)
    return probs[:, 1].tolist()  # 返回 yes 的概率

# === 主处理流程 ===
def rerank_batch(task, questions, recalls, batch_size=8):
    results = []
    for question, recall_list in zip(questions, recalls):
        pairs = [format_instruction(task, question, rec['article']['content']) for rec in recall_list]
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            inputs = process_inputs(batch_pairs)
            scores += compute_logits(inputs)
        for rec, s in zip(recall_list, scores):
            rec['rerank_score'] = s
        results.append(recall_list)
    return results

def main(input_path, output_path, instruction="Given a web search query, retrieve relevant passages that answer the query"):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in tqdm(data, desc="Scoring"):
        questions = []
        recall_lists = []
        for turn in item["conversation"]:
            questions.append(turn["question"]) # 采用question 字段还是原来的 query 字段
            recall_lists.append(turn["recall"])
        reranked = rerank_batch(instruction, questions, recall_lists)
        for turn, scored in zip(item["conversation"], reranked):
            turn["recall"] = scored

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_path = "input.json"
    output_path = "output_reranked.json"
    main(input_path, output_path)
