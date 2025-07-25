#!/bin/zsh

cd /home/liuxj25/LawLLM/LLaMA-Factory

FORCE_TORCHRUN=1 llamafactory-cli train ccir/qwen3-32b-lora-sft-ds3.yaml 