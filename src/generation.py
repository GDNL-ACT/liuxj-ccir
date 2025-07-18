import os
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
    ep = EvaluatorPipeline()
    gp = GeneratorPipeline(
        model_type=args.generate_model_type,
        config={
            "model_type": args.generate_model_type,
            "model_path": args.generate_model_path
        }
    )
    gp.run_generator(
        raw_data_path=args.raw_conversation_path,
        retrieval_data_path=retriever_output_path,
        max_retries=args.max_retries,
        max_parallel=args.max_parallel,
        top_n=args.top_n,
        batch_size=args.batch_size
    )
    if(args.enable_evaluate):
        gen_metrics = ep.run_evaluator(
            eval_type="generation",
            metrics=["bert-score","keyword_accuracy"],
            data_path=args.raw_conversation_path,
            response_file=generator_output_file
        )
        print("生成评估结果:", gen_metrics)

    print(args)
    
if __name__ == "__main__":
    main()

