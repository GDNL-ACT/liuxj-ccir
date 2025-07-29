import json
import math
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from vllm.inputs.data import TokensPrompt

def format_instruction(instruction, query, doc):
    return [
        {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. The answer must be \"yes\" or \"no\"."},
        {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
    ]

def process_inputs(pairs, instruction, max_length, suffix_tokens, tokenizer):
    messages = [format_instruction(instruction, q, d) for q, d in pairs]
    tokenized = [
        tokenizer.apply_chat_template(m, tokenize=True, add_generation_prompt=False, enable_thinking=False)
        for m in messages
    ]
    tokenized = [m[:max_length] + suffix_tokens for m in tokenized]
    return [TokensPrompt(prompt_token_ids=m) for m in tokenized]

def compute_logits(model, messages, sampling_params, true_token, false_token):
    outputs = model.generate(messages, sampling_params, use_tqdm=False)
    scores = []
    for output in outputs:
        logits = output.outputs[0].logprobs[-1]
        true_logit = logits.get(true_token, None)
        false_logit = logits.get(false_token, None)
        true_score = math.exp(true_logit.logprob) if true_logit else 1e-4
        false_score = math.exp(false_logit.logprob) if false_logit else 1e-4
        prob = true_score / (true_score + false_score)
        scores.append(prob)
    return scores

def main():
    input_path = "/home/liuxj25/LawLLM/CCIR/eval/vllm/output/0.0/A_retrieval(11.48).json"
    output_path = "/home/liuxj25/LawLLM/CCIR/eval/vllm/output/0.0/A_retrieval(11.48)_reranked.json"
    rerank_model = "/home/liuxj25/LawLLM/CCIR/models/Qwen3-reranker-8B"
    
    instruction = "Given a legal consultation query, rank relevant legal articles based on how well they support the query."

    # Load data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(rerank_model)
    model = LLM(model=rerank_model,
                tensor_parallel_size=torch.cuda.device_count(),
                max_model_len=8192,
                enable_prefix_caching=True)

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    max_input_len = 8192 - len(suffix_tokens)

    true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
    false_token = tokenizer("no", add_special_tokens=False).input_ids[0]

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=20,
        allowed_token_ids=[true_token, false_token]
    )

    for dialog in tqdm(data, desc="Reranking"):
        for turn in dialog["conversation"]:
            query = turn["question"]
            articles = turn.get("recall", [])

            pairs = [(query, a["article"]["name"] + "：" + a["article"]["content"]) for a in articles]
            if not pairs:
                continue

            prompts = process_inputs(pairs, instruction, max_input_len, suffix_tokens, tokenizer)
            scores = compute_logits(model, prompts, sampling_params, true_token, false_token)

            for i in range(len(articles)):
                articles[i]["score"] = scores[i]

            articles.sort(key=lambda x: x["score"], reverse=True)
            turn["recall"] = articles  # 覆盖原 recall

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    destroy_model_parallel()

if __name__ == "__main__":
    main()
