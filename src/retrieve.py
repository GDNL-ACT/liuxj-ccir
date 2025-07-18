import os
import argparse
from pipeline import GeneratorPipeline, ProcessorPipeline, RetrieverPipeline, EvaluatorPipeline
import shutil
import logging

def parse_args():
    parser = argparse.ArgumentParser(description="LexRAG Pipeline")

    # 模型与策略参数
    parser.add_argument('--retriever_model_type', type=str) # 'Qwen2-1.5B','BGE-base-zh','local'
    parser.add_argument('--retriever_model_path', type=str, required=True)
    parser.add_argument('--process_type', type=str, default='prefix_question') #['current_question', 'prefix_question', 'prefix_question_answer', 'suffix_question', 'rewrite_question']
    parser.add_argument("--processor_model_type", type=str, default="local")
    parser.add_argument("--processor_model_path", type=str)    
    # 路径参数
    parser.add_argument('--raw_conversation_path', type=str, default='data/dataset.json')
    parser.add_argument('--law_corpus_path', type=str, default='data/law_library.jsonl')

    parser.add_argument('--enable_evaluate', action='store_true', help='是否执行评估')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 构建文件路径
    processor_output_path = f"data/dataset_{args.process_type}.jsonl"
    retriever_output_path = f"data/retrieval/res/retrieval_{args.retriever_model_type}_{args.process_type}.jsonl"
    
    # 1. 预处理
    if args.process_type.startswith("rewrite_question"):
        pp = ProcessorPipeline(
                config={
                    "model_type": args.processor_model_type,
                    "model_path": args.processor_model_path
                }
            )
    else:
        pp = ProcessorPipeline()
    pp.run_processor(
        process_type=args.process_type,
        original_data_path=args.raw_conversation_path,
        output_path=processor_output_path
    )

    # 2. 检索
    rp = RetrieverPipeline()
    rp.run_retriever(
        model_type=args.retriever_model_type,
        model_name=args.retriever_model_path,
        faiss_type="FlatIP",
        process_type=args.process_type,
        question_file_path=processor_output_path,
        law_path=args.law_corpus_path
    )
    ep = EvaluatorPipeline()
    if(args.enable_evaluate):
        # 3. 评估（检索）
        ret_metrics = ep.run_evaluator(
            eval_type="retrieval",
            metrics=["ndcg"],
            results_path=retriever_output_path,
            k_values=[5]
        )
        print("检索评估结果:", ret_metrics)

    print(args)

if __name__ == "__main__":
    main()

