import json
import time
import httpx
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from zhipuai import ZhipuAI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Rewriter:
    def __init__(self, config, max_retries, max_parallel, batch_size):
        self.config = config
        self.max_retries = max_retries
        self.max_parallel = max_parallel
        self.batch_size = batch_size
        self.model_name = self.config.get("model_name")
        
        if self.config.get("model_type") == "openai":
            self.client = OpenAI(
                base_url=self.config.get("api_base"),
                api_key=self.config.get("api_key"),
                http_client=httpx.Client(
                    base_url=self.config.get("api_base"),
                    follow_redirects=True,
                ),
            )
        elif self.config.get("model_type") == "zhipu":
            self.client = ZhipuAI(api_key=self.config.get("api_key"))
        elif self.config.get("model_type") in ["qwen","llama"]:
            self.client = OpenAI(
                base_url=self.config.get("api_base"), 
                api_key=self.config.get("api_key")
            )
        elif self.config.get("model_type") == "local":
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.get("model_path"), trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.get("model_path"),
                torch_dtype=torch.float16,               
                device_map="auto",                   
                trust_remote_code=True
            )
            self.model.eval()
        
        self.results = {}
        self.failed_ids = set()

    def generate_prompt(self, history, question):
        if not history.strip() or history == "无历史对话":
            return question  
        return f"""给定以下对话(包括历史对话和当前问题)，请将用户的当前问题改写为一个独立的问题，使其无需依赖对话历史即可理解用户的意图。
在改写用户的当前问题时，请避免不必要的措辞修改或引入对话中未提及的新术语或概念。改写应尽可能接近用户当前问题的结构和含义。

历史对话：
{history}

当前问题：{question}

请输出改写结果(只输出结果即可)："""

    def process_single(self, data_id, conv_idx, history, question):
        if not history.strip() or history == "无历史对话":
            return (data_id, conv_idx, question)
        unique_id = f"{data_id}_{conv_idx}"
        prompt = self.generate_prompt(history, question)
        for attempt in range(self.max_retries):
            try:
                if self.config.get("model_type") == "local":
                    messages = [{"role": "user", "content": prompt}]
                    prompt_chat = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                    model_inputs = self.tokenizer(
                        prompt_chat, 
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        return_attention_mask=True
                    )
                    model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids=model_inputs["input_ids"],
                            attention_mask=model_inputs["attention_mask"],
                            do_sample=False,
                            max_new_tokens=256
                    )
                    reworded = self.tokenizer.decode(
                        outputs[0][len(model_inputs["input_ids"][0]):], 
                        skip_special_tokens=True
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{
                            "role": "user",
                            "content": prompt
                        }],
                        temperature=0.0,
                        stream=False
                    )
                    reworded = response.choices[0].message.content.strip()

                reworded = reworded.replace('"', '').replace("```", "").strip()
                return (data_id, conv_idx, reworded)
            
            except Exception as e:
                print(f"[{unique_id}] Attempt {attempt+1} failed: {str(e)}")
                time.sleep(2 ** attempt)
        
        self.failed_ids.add(unique_id)
        return (data_id, conv_idx, question) 

    def batch_process(self, original_data, output_path):
        with open(original_data, "r", encoding="utf-8") as f:
            original_data = json.load(f)
        
        tasks = []
        for data in original_data:
            data_id = data["id"]
            for conv_idx, conv in enumerate(data["conversation"]):
                history = "\n".join(
                    [f"用户：{c['user']}\n助理：{c['assistant']}" 
                     for c in data["conversation"][:conv_idx]]
                )
                if not history.strip():
                    history = "无历史对话"
                tasks.append((
                    data_id,
                    conv_idx,
                    history,
                    conv["user"]
                ))
        
        completed_count = 0

        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = []
            
            for task in tasks:
                future = executor.submit(self.process_single, *task)
                futures.append(future)
            
            with tqdm(total=len(tasks), desc="Processing") as pbar:
                for future in as_completed(futures):
                    data_id, conv_idx, reworded = future.result()
                    unique_id = f"{data_id}_{conv_idx}"
                    self.results[unique_id] = reworded
                    completed_count += 1
                    pbar.update(1)
                    if completed_count % self.batch_size == 0:
                        self.generate_output_file(original_data, self.results, output_path)
        
        self.generate_output_file(original_data, self.results, output_path)
        print(f"Process completed. Failed: {len(self.failed_ids)}")

    def generate_output_file(self, original_data, results, output_path):
        output_lines = []
        for data in original_data:
            data_id = data["id"]
            for conv_idx, conv in enumerate(data["conversation"]):
                unique_id = f"{data_id}_{conv_idx}"
                conv["question"] = {
                    "type": "llm_question",
                    "content": results.get(unique_id, "")  
                }
            output_lines.append(json.dumps(data, ensure_ascii=False))
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines) + "\n")
