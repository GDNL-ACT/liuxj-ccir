from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import shutil
import os

def _load_lora_chat(base_path, lora_path, save_path=None):
    if save_path is not None:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)  # 递归删除旧目录
        os.makedirs(save_path, exist_ok=True)
        
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype="auto",              
        device_map="auto", 
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(
        base_path, 
        trust_remote_code=True, 
        padding_side="left"
    )
    
    if save_path is not None:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Merged model saved to {save_path}")

    return model, tokenizer

def _load_lora_emb(base_path, lora_path, save_path=None):
    if save_path is not None:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)  # 递归删除旧目录
        os.makedirs(save_path, exist_ok=True)
        
    base_model = AutoModel.from_pretrained(
        base_path,
        torch_dtype="auto",              
        device_map="auto", 
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(
        base_path, 
        trust_remote_code=True, 
        padding_side="left"
    )
    
    if save_path is not None:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Merged model saved to {save_path}")
    
    return model, tokenizer

if __name__ == "__main__":
    # # 加载聊天模型并保存
    # model, tokenizer = _load_lora_chat(
    #     base_path="/home/liuxj25/LawLLM/CCIR/models/Qwen3-32B",
    #     lora_path="/home/liuxj25/LawLLM/CCIR/train/generation/checkpoints/Qwen3-32B-v1",
    #     save_path="/home/liuxj25/LawLLM/CCIR/eval/models/Qwen3chat"
    # )

    # 加载嵌入模型并保存
    model, tokenizer = _load_lora_emb(
        base_path="/home/liuxj25/LawLLM/CCIR/models/Qwen3-embedding-8B",
        lora_path="/home/liuxj25/LawLLM/CCIR/train/retrieval/checkpoints/Qwen3-Embedding8B-v3",
        save_path="/home/liuxj25/LawLLM/CCIR/eval/models/Qwen3embedding"
    )