#!/bin/bash

# 设置默认路径
BASE_DIR="/home/liuxj25/LawLLM/CCIR"
MODEL_PATH="$BASE_DIR/models/Qwen3-32B"
EMB_MODEL_PATH="$BASE_DIR/models/Qwen3-embedding-8B"
LAW_LIB_PATH="$BASE_DIR/data/law_library.jsonl"
QUESTION_FILE="$BASE_DIR/eval/vllm/data/dataset1_question.json"
OUTPUT_DIR="$BASE_DIR/eval/vllm/data/generation(vllm)/"

TENSOR_PARALLEL_SIZE=4

mkdir -p "$OUTPUT_DIR"
cd  "$BASE_DIR/eval/vllm"

# Step 1: Generate pseudo answers
echo "\n[Step 1/4] Generating pseudo answers..."
python pseudo_vllm.py \
    --model_path "$MODEL_PATH" \
    --data_path "$QUESTION_FILE" \
    --output_path "$OUTPUT_DIR/A_pseudo.jsonl" \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --batch_size 128

# Step 2: Rewrite questions
echo  "\n[Step 2/4] Rewriting questions..."
python rewriter_vllm.py \
    --model_path "$MODEL_PATH" \
    --data_path "$OUTPUT_DIR/A_pseudo.jsonl" \
    --output_path "$OUTPUT_DIR/A_rewritten.json" \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --batch_size 128

# # Step 3: Retrieve relevant laws
# echo  "\n[Step 3/4] Retrieving relevant laws..."
# python retriever_vllm.py \
#     --model_path "$EMB_MODEL_PATH" \
#     --input_path "$OUTPUT_DIR/A_rewritten.json" \
#     --law_path "$LAW_LIB_PATH" \
#     --output_path "$OUTPUT_DIR/A_retrieval.json" \
#     --batch_size 64 \
#     --top_k 5

# # Step 4: Generate final answers
# echo "\n[Step 4/4] Generating final answers..."
# python generator_vllm.py \
#     --model_path "$MODEL_PATH" \
#     --input_path "$OUTPUT_DIR/A_retrieval.json" \
#     --output_path "$OUTPUT_DIR/A_output.json" \
#     --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
#     --batch_size 128 \
#     --max_article 3
