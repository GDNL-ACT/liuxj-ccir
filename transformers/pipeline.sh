#!/bin/bash
cd /home/liuxj25/LawLLM/CCIR/eval

# python pipeline.py \
#   --top_k 5 \
#   --retriever_batch_size 64 \
#   --generator_batch_size 16 \
#   --retriever_model_path  /home/liuxj25/LawLLM/CCIR/eval/models/20250726_0537 \
#   --generator_model_path /home/liuxj25/LawLLM/CCIR/models/Qwen3-32B \
#   --raw_data_path ../data/question_A.json \
#   --law_path ../data/tmp/law_test.jsonl \
#   --pseudo_output_path ./output/pseudo.jsonl \
#   --processor_output_path ./output/queries.json \
#   --retrieval_output_path ./output/retrieval.json \
#   --generation_output_path ./output/output.json


python pipeline.py \
  --top_k 5 \
  --retriever_batch_size 64 \
  --generator_batch_size 16 \
  --retriever_model_path /home/liuxj25/LawLLM/CCIR/models/Qwen3-embedding-8B \
  --retriever_lora_path /home/liuxj25/LawLLM/CCIR/train/retrieval/checkpoints/Qwen3-Embedding8B-v2 \
  --generator_model_path /home/liuxj25/LawLLM/CCIR/models/Qwen3-32B \
  --raw_data_path ../data/question_A.json \
  --law_path ../data/law_library.jsonl \
  --pseudo_output_path ./output/2.0.0/pseudo.jsonl \
  --processor_output_path ./output/2.0.0/queries.json \
  --retrieval_output_path ./output/2.0.0/retrieval.json \
  --generation_output_path ./output/2.0.0/output.json
  
python pipeline.py \
  --top_k 5 \
  --retriever_batch_size 64 \
  --generator_batch_size 16 \
  --retriever_model_path /home/liuxj25/LawLLM/CCIR/models/Qwen3-embedding-8B \
  --generator_model_path /home/liuxj25/LawLLM/CCIR/models/Qwen3-32B \
  --raw_data_path ../data/question_A.json \
  --law_path ../data/law_library.jsonl \
  --pseudo_output_path ./output/3.0.0/pseudo.jsonl \
  --processor_output_path ./output/3.0.0/queries.json \
  --retrieval_output_path ./output/3.0.0/retrieval.json \
  --generation_output_path ./output/3.0.0/output.json
