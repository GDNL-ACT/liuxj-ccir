#!/bin/zsh


# 安装LLaMAFactory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation

# 训练运行脚本
cd /home/liuxj25/LawLLM/LLaMA-Factory

FORCE_TORCHRUN=1 llamafactory-cli train ccir/qwen3-32b-lora-sft-ds3.yaml 

FORCE_TORCHRUN=1 llamafactory-cli train ccir/qwen3-32b-lora-ppo-ds3.yaml 