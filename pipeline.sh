#!/bin/bash
cd /home/liuxj25/LawLLM/CCIR/eval

python pipeline.py \
  --top_k 5 \
  --faiss_type FlatIP \
  --retriever_batch_size 32 \
  --generator_batch_size 16 \
  --retriever_model_path /home/liuxj25/LawLLM/CCIR/models/Qwen3-embedding-8B \
  --generator_model_path /home/liuxj25/LawLLM/CCIR/models/Qwen3-32B \
  --raw_data_path ../data/question_A.json \
  --law_path ../data/law_library.jsonl \
  --pseudo_output_path ./output/pseudo.jsonl \
  --processor_output_path ./output/queries.json \
  --retrieval_output_path ./output/retrieval.json \
  --generation_output_path ./output/output.json

python pipeline.py \
  --top_k 5 \
  --faiss_type FlatIP \
  --retriever_batch_size 32 \
  --generator_batch_size 16 \
  --retriever_model_path /home/liuxj25/LawLLM/CCIR/models/Qwen3-embedding-8B \
  --lora_path /home/liuxj25/LawLLM/CCIR/train/retrieval/checkpoints/finetuned-Qwen3-Embedding8B-36bsz \
  --generator_model_path /home/liuxj25/LawLLM/CCIR/models/Qwen3-32B \
  --raw_data_path ../data/question_A.json \
  --law_path ../data/law_library.jsonl \
  --pseudo_output_path ./output/pseudo.jsonl \
  --processor_output_path ./output/queries.json \
  --retrieval_output_path ./output/retrieval.json \
  --generation_output_path ./output/output.json

# python pipeline.py \
#   --top_k 5 \
#   --faiss_type FlatIP \
#   --process_type "current_question" \
#   --retriever_batch_size 32 \
#   --generator_batch_size 4 \
#   --retriever_model_path /home/liuxj25/LawLLM/CCIR/models/Qwen3-embedding-8B \
#   --lora_path /home/liuxj25/LawLLM/CCIR/train/retrieval/checkpoints/finetuned-Qwen3-Embedding8B-36bsz \
#   --generator_model_path /home/liuxj25/LawLLM/CCIR/models/Qwen3-32B \
#   --raw_data_path ../data/question_A.json \
#   --law_path ../data/law_library.jsonl \
#   --pseudo_output_path ./output/pseudo.jsonl \
#   --processor_output_path ./output/queries.json \
#   --retrieval_output_path ./output/retrieval.json \
#   --generation_output_path ./output/output.json

