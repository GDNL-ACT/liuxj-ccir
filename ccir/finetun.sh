#!/bin/zsh
cd /home/liuxj25/LawLLM/Tevatron/tevatron

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export NCCL_SHM_DISABLE=1

deepspeed --include localhost:0,1,2,3 --master_port 50000 --module tevatron.retriever.driver.train \
  --do_train \
  --lora \
  --lora_r 8 \
  --lora_alpha 16 \
  --output_dir ccir/checkpoints/Qwen3-emb-8b-8bsz4 \
  --model_name_or_path /home/liuxj25/LawLLM/CCIR/models/Qwen3-embedding-8B \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj \
  --deepspeed deepspeed/ds_zero3_config.json \
  --dataset_path  ccir/data/data_whn3.jsonl \
  --corpus_path ccir/data/law_libraty.jsonl \
  --query_prefix "Instruct: 根据用户的法律问题，从法律条文中检索出最相关的一条。\n问题：" \
  --passage_prefix "" \
  --bf16 \
  --pooling last \
  --padding_side left \
  --normalize \
  --temperature 0.01 \
  --train_group_size 4 \
  --gradient_checkpointing \
  --overwrite_output_dir \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --query_max_len 256 \
  --passage_max_len 512 \
  --num_train_epochs 5 \
  --save_steps 500 \
  --logging_steps 100 \
  --per_device_train_batch_size 8 \
  --attn_implementation eager

# deepspeed --include localhost:0,1 --master_port 60000 --module tevatron.retriever.driver.train \
#   --do_train \
#   --lora \
#   --lora_r 8 \
#   --lora_alpha 16 \
#   --output_dir ccir/checkpoints/Qwen3-emb-8b-8bsz \
#   --model_name_or_path /home/liuxj25/LawLLM/CCIR/models/Qwen3-embedding-8B \
#   --lora_target_modules q_proj,k_proj,v_proj,o_proj \
#   --deepspeed deepspeed/ds_zero3_config.json \
#   --dataset_path  ccir/data/data_whn3.jsonl \
#   --corpus_path ccir/data/law_libraty.jsonl \
#   --query_prefix "Instruct: 根据用户的法律问题，从法律条文中检索出最相关的一条。\n问题：" \
#   --passage_prefix "" \
#   --bf16 \
#   --pooling last \
#   --padding_side left \
#   --normalize \
#   --temperature 0.01 \
#   --train_group_size 4 \
#   --gradient_checkpointing \
#   --overwrite_output_dir \
#   --gradient_accumulation_steps 1 \
#   --learning_rate 1e-4 \
#   --query_max_len 256 \
#   --passage_max_len 512 \
#   --num_train_epochs 5 \
#   --save_steps 500 \
#   --logging_steps 100 \
#   --per_device_train_batch_size 8 \
#   --attn_implementation eager