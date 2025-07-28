#!/bin/zsh
cd /home/liuxj25/LawLLM/Tevatron/tevatron

deepspeed --include localhost:0,1,2,3 --master_port 51234 --module tevatron.retriever.driver.train \
  --do_train \
  --lora \
  --lora_r 8 \
  --lora_alpha 16 \
  --output_dir ccir/checkpoints/finutuned-Qwen3-embedding-8b \
  --model_name_or_path /home/liuxj25/LawLLM/CCIR/models/Qwen3-embedding-8B \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj \
  --deepspeed deepspeed/ds_zero3_config.json \
  --dataset_path  ccir/data/data_whn3.jsonl \
  --corpus_path ccir/data/law_library.jsonl \
  --query_prefix "" \
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
  --query_max_len  1024 \
  --passage_max_len 1024 \
  --num_train_epochs 10 \
  --save_steps 500 \
  --logging_steps 100 \
  --per_device_train_batch_size 8 \
  --attn_implementation eager