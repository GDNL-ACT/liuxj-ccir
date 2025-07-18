#!/bin/bash
cd /home/liuxj25/LawLLM/LexRAG

# 修改process_type, processor_model_path, retriever_model_type, retriever_model_path
python src/retrieve.py \
  --process_type rewrite_question_4B \
  --processor_model_type local \
  --processor_model_path /home/liuxj25/LawLLM/CCIR/models/Qwen3-4B \
  --retriever_model_type Qwen3-embedding-0.6B \
  --retriever_model_path /home/liuxj25/LawLLM/CCIR/models/Qwen3-embedding-0.6B \
  --raw_conversation_path data/dataset.json \
  --law_corpus_path data/law_library.jsonl \
  --enable_evaluate
