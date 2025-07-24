import gc
import torch
import logging
import argparse
from processor import Processor
from retriever import Retriever
from generator import Generator
from pseudo import PseudoAnswerGenerator

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def main(args):
    pseudoAnswerGenerator = PseudoAnswerGenerator(model_path=args.generator_model_path)
    pseudoAnswerGenerator.run(data_path=args.raw_data_path, output_path=args.pseudo_output_path,batch_size=args.generator_batch_size)
    del pseudoAnswerGenerator
    torch.cuda.empty_cache()
    gc.collect()

    processor_rewrite = Processor("rewrite_question_eval", model_path=args.generator_model_path, batch_size=args.generator_batch_size)
    processor_rewrite.run(args.pseudo_output_path, args.processor_output_path)
    del processor_rewrite
    torch.cuda.empty_cache()
    gc.collect()

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

    generator = Generator(model_path=args.generator_model_path,)
    generator.run(
        input_path=args.retrieval_output_path,
        output_path=args.generation_output_path,
        batch_size=args.generator_batch_size
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Framework Pipeline")

    # Processor params
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
