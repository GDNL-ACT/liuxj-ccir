import os
import argparse
from pipeline import GeneratorPipeline, ProcessorPipeline, RetrieverPipeline, EvaluatorPipeline
import shutil
import logging

def parse_args():
    parser = argparse.ArgumentParser(description="LexRAG Pipeline")

    # 模型与策略参数
    parser.add_argument('--generate_model_type', type=str, default='local')
    parser.add_argument('--generate_model_path', type=str, required=True)
    parser.add_argument('--retriever_model_type', type=str, default='local') # 'Qwen2-1.5B','BGE-base-zh','local'
    parser.add_argument('--retriever_model_path', type=str, required=True)
    parser.add_argument('--process_type', type=str, default='prefix_question',
                        choices=['current_question', 'prefix_question', 'prefix_question_answer', 'suffix_question', 'rewrite_question'])

    parser.add_argument('--enable_evaluate', action='store_true', help='是否执行评估')

    # 路径参数
    parser.add_argument('--raw_conversation_path', type=str, default='data/dataset.json')
    parser.add_argument('--law_corpus_path', type=str, default='data/law_library.jsonl')
    parser.add_argument('--generator_response_file', type=str, default='data/generated_responses.jsonl')

    # 生成参数
    parser.add_argument('--enable_generation', action='store_true', help='是否执行生成')
    parser.add_argument('--max_retries', type=int, default=5)
    parser.add_argument('--max_parallel', type=int, default=4)
    parser.add_argument('--top_n', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=20)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 构建文件路径
    processor_output_path = f"data/dataset_{args.process_type}.jsonl"
    retriever_output_path = f"data/retrieval/res/retrieval_{args.retriever_model_type}_{args.process_type}.jsonl"
    
    ## 1. 预处理
    # pp = ProcessorPipeline(
    #         config={
    #             "model_type": "local",
    #             "model_path": "/home/liuxj25/LawLLM/CCIR/models/Qwen3-32B"
    #         }
    #     )
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
        # print("检索评估结果:", ret_metrics)

    # 4. 生成
    if args.enable_generation:
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
            # 5. 评估（生成）
            gen_metrics = ep.run_evaluator(
                eval_type="generation",
                metrics=["bleu", "rouge", "keyword_accuracy"],
                data_path=args.raw_conversation_path,
                response_file=args.generator_response_file
            )
            # print("生成评估结果:", gen_metrics)

        move_file_safely(args.generator_response_file,"data/generation")

    print(args)

def move_file_safely(source_path, target_dir):
    filename = os.path.basename(source_path)
    target_path = os.path.join(target_dir, filename)

    if os.path.exists(target_path):
        logging.warning(f"responses文件已存在，跳过移动操作：{target_path}")
        return False
    else:
        os.makedirs(target_dir, exist_ok=True)
        shutil.move(source_path, target_path)
        logging.info(f"responses文件成功移动到：{target_path}")
        return True
    
if __name__ == "__main__":
    main()

