#!/bin/bash
cd /home/liuxj25/LawLLM/LexRAG

# 修改generate_model_type 和 generate_model_path 为[Qwen3-32B、Qwen3-8B]的路径，即可测试0shot下的生成
python src/generation.py \
  --generate_model_type Qwen3-8B-gold \
  --generate_model_path "/home/liuxj25/LawLLM/CCIR/models/Qwen3-8B" \
  --retriever_model_type Qwen3-embedding-8B \
  --process_type current_question \
  --raw_conversation_path "data/dataset.json" \
  --top_n 5 \
  --max_retries 5 \
  --max_parallel 4 \
  --batch_size 20 \
  --enable_evaluate