#!/bin/bash

cd /home/liuxj25/LawLLM/LexRAG

python src/main.py \
  --generate_model_type local \
  --generate_model_path "/home/liuxj25/LawLLM/CCIR/models/chatglm3" \
  --retriever_model_type local_Qwen3-embedding-8B \
  --retriever_model_path "/home/liuxj25/LawLLM/CCIR/models/Qwen3-embedding-8B" \
  --process_type current_question \
  --raw_conversation_path "data/dataset.json" \
  --law_corpus_path "data/law_library.jsonl" \
  --generator_response_file "data/generated_responses.jsonl" \
  --top_n 5 \
  --max_retries 5 \
  --max_parallel 4 \
  --batch_size 20 \
  --enable_evaluate
  # --enable_generation \
