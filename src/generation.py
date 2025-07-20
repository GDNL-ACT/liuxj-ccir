import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
from pipeline import GeneratorPipeline, ProcessorPipeline, RetrieverPipeline, EvaluatorPipeline
import shutil
import logging

def parse_args():
    parser = argparse.ArgumentParser(description="LexRAG Pipeline")

    # 模型与策略参数
    parser.add_argument('--generate_model_type', type=str, default='Qwen3-8B-gold')
    parser.add_argument('--generate_model_path', type=str,default='/home/liuxj25/LawLLM/CCIR/models/Qwen3-8B')
    parser.add_argument('--process_type', type=str, default='current_question',
                        choices=['current_question', 'prefix_question', 'prefix_question_answer', 'suffix_question', 'rewrite_question'])
    parser.add_argument('--retriever_model_type', type=str,default='Qwen3-embedding-8B')
    
    parser.add_argument('--enable_evaluate', action='store_true', help='是否执行评估')

    # 路径参数
    parser.add_argument('--raw_conversation_path', type=str, default='data/dataset.json')

    # 生成参数
    parser.add_argument('--max_retries', type=int, default=5)
    parser.add_argument('--max_parallel', type=int, default=4)
    parser.add_argument('--top_n', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_args()
    # 构建文件路径
    retriever_output_path = f"data/retrieval/res/retrieval_{args.retriever_model_type}_{args.process_type}.jsonl"
    generator_output_file = f"data/generation/responses_{args.generate_model_type}.jsonl"
    # gp = GeneratorPipeline(
    #     model_type=args.generate_model_type,
    #     config={
    #         "model_type": args.generate_model_type,
    #         "model_path": args.generate_model_path
    #     }
    # )
    # gp.run_generator(
    #     raw_data_path=args.raw_conversation_path,
    #     retrieval_data_path=retriever_output_path,
    #     max_retries=args.max_retries,
    #     max_parallel=args.max_parallel,
    #     top_n=args.top_n,
    #     batch_size=args.batch_size
    # )
    ep = EvaluatorPipeline()
    if(args.enable_evaluate):
        gen_metrics = ep.run_evaluator(
            eval_type="generation",
            metrics=["bert-score","keyword_accuracy"],
            data_path=args.raw_conversation_path,
            response_file=generator_output_file
        )

    def save_avg_metrics(result_dict, output_path):
        total_f1 = 0.0
        total_acc = 0.0
        count = 0

        for item in result_dict.values():
            total_f1 += item.get("bert-f1", 0.0)
            total_acc += item.get("keyword_accuracy", 0.0)
            count += 1

        avg_f1 = round(total_f1 / count, 4)
        avg_acc = round(total_acc / count, 4)

        result_summary = {
            "response_path": generator_output_file,
            "avg_bert_f1": avg_f1,
            "avg_keyword_accuracy": avg_acc,
            "score" : 50 * avg_f1 + 50 * avg_acc
        }

        # 追加写入文件，每次写一行 JSON
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_summary, ensure_ascii=False) + "\n")
    save_avg_metrics(gen_metrics,"data/generation/report.jsonl")
    print(args)

import json


if __name__ == "__main__":
    main()

