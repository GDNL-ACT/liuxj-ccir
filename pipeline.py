import gc
import torch
import logging
import argparse
from processor import Processor
from retriever import Retriever
from generator import PromptBuilder, Generator
from pseudo import PseudoAnswerGenerator

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def main(args):
    if args.process_type == "rewrite_question":
        promptPseudo = PromptBuilder(
            system_prompt=(
                "你是一名精通中国法律的专业法律顾问，正在模拟多轮法律问答场景中的助手回复。"
                "请根据用户的每一轮提问内容以及历史的回答，生成一条语义完整、逻辑合理、风格正式的法律解答，用于构造对话历史。"
                "要求如下：\n"
                "1. 回答应围绕问题核心展开，表达清晰，具有法律常识基础和推理合理性；\n"
                "2. 可使用通用的法律术语，但不需要引用具体法律条文；\n"
                "3. 保持正式、客观、专业的语气，避免模糊表达和闲聊式语言；\n"
                "4. 回答应简明扼要，结构清晰，避免冗长；\n"
                "5. 仅输出本轮问题的回答内容，不输出任何多余说明或注释。\n\n"
                "现在请开始生成回复："
            )
        )
        # promptPseudo = PromptBuilder()
        pseudoAnswerGenerator = PseudoAnswerGenerator(model_path=args.generator_model_path, prompt_builder=promptPseudo)
        pseudoAnswerGenerator.run(data_path=args.raw_data_path, output_path=args.pseudo_output_path,batch_size=args.generator_batch_size)
        del pseudoAnswerGenerator
        torch.cuda.empty_cache()
        gc.collect()

        processor_rewrite = Processor("rewrite_question", model_path=args.generator_model_path, batch_size=args.generator_batch_size)
        processor_rewrite.run(args.pseudo_output_path, args.processor_output_path)
        del processor_rewrite
        torch.cuda.empty_cache()
        gc.collect()
    else:
        processor = Processor(args.process_type)
        processor.run(args.raw_data_path, args.processor_output_path)

    retriever = Retriever(
        model_path=args.retriever_model_path,
        lora_path=args.lora_path,
        faiss_type=args.faiss_type,
        batch_size=args.retriever_batch_size,
        index_path=args.retriever_index_path
    )
    retriever.run(
        input_path=args.processor_output_path,
        law_path=args.law_path,
        output_path=args.retrieval_output_path,
        top_k=args.top_k
    )
    del retriever
    torch.cuda.empty_cache()
    gc.collect()

    prompt_builder = PromptBuilder()
    generator = Generator(
        model_path=args.generator_model_path,
        prompt_builder=prompt_builder
    )
    generator.run(
        input_path=args.retrieval_output_path,
        output_path=args.generation_output_path,
        batch_size=args.generator_batch_size
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Framework Pipeline")

    # Processor params
    parser.add_argument("--process_type", type=str, default="current_question")
    parser.add_argument("--raw_data_path", type=str, default="/home/liuxj25/LawLLM/CCIR/data/question_A.json")
    parser.add_argument("--processor_output_path", type=str, default="./output/tmp.jsonl")
    parser.add_argument("--pseudo_output_path", type=str, default="./output/pseudo.jsonl")

    # Retriever params
    parser.add_argument("--retriever_model_path", type=str, default="/home/liuxj25/LawLLM/CCIR/models/gte_Qwen2")
    parser.add_argument("--lora_path", type=str)
    parser.add_argument("--retriever_index_path", type=str)
    parser.add_argument("--faiss_type", type=str, default="FlatIP")
    parser.add_argument("--retriever_batch_size", type=int, default=32)
    parser.add_argument("--law_path", type=str, default="/home/liuxj25/LawLLM/CCIR/data/law_library.jsonl")
    parser.add_argument("--retrieval_output_path", type=str, default="./output/retmp.json")
    parser.add_argument("--top_k", type=int, default=5)

    # Generator params
    parser.add_argument("--generator_model_path", type=str, default="/home/liuxj25/LawLLM/CCIR/models/Qwen3-4B")
    parser.add_argument("--generation_output_path", type=str, default="./output.json")
    parser.add_argument("--generator_batch_size", type=int, default=8)

    args = parser.parse_args()
    main(args)
    print(args)
