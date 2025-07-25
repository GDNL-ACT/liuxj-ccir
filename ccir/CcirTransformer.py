import json
import argparse
from collections import defaultdict

def convert_corpus_format(input_path: str, output_path: str):
    """
    将原始语料库转换为 Tevatron 格式
    输入格式：{"id": ..., "name": ..., "content": ...}
    输出格式：{"docid": ..., "title":... , "text": ...}
    """
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        
        for line in fin:
            item = json.loads(line)
            converted = {
                "docid": item.get("id"),
                "title": item.get("name", ""),
                "text": item.get("content", "")
            }
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")

def build_title_index(corpus_path):
    """
    构建 title -> (docid, text) 的映射
    """
    title2meta = {}
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            title2meta[item['title']] = (item['docid'], item['text'].strip())
    return title2meta

def convert_training_data_by_title(train_path, output_path, title2meta):
    from collections import defaultdict
    missing_counter = defaultdict(int)

    def extract_title_from_text(text):
        return (text.split("：")[0])

    with open(train_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for idx, line in enumerate(fin):
            item = json.loads(line.strip())
            query_id = idx
            query = item['query'].strip()

            def convert_passages(passages, label):
                results = []
                for text in passages:
                    title_guess = extract_title_from_text(text)
                    if title_guess in title2meta:
                        docid, full_text = title2meta[title_guess]
                        results.append({
                            "docid": docid,
                            "title": title_guess,
                            "text": full_text
                        })
                    else:
                        docid = f"missing_{hash(text)}"
                        full_text = text
                        missing_counter[label] += 1
                    
                return results

            positives = convert_passages(item.get("pos", []), label="pos")
            negatives = convert_passages(item.get("negs", []), label="neg")

            if positives and negatives:
                new_item = {
                "query_id": query_id,
                "query": query,
                "positive_passages": positives,
                "negative_passages": negatives
                }
                fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")
    
    return missing_counter
def main():
    raw_corpus = "/home/liuxj25/LawLLM/CCIR/data/law_library.jsonl"
    corpus_output = "data/law_libraty.jsonl"
    raw_train = "/home/liuxj25/LawLLM/CCIR/train/retrieval/data/data2_whn3.jsonl"
    train_output = "data/data2_whn3.jsonl"

    # convert_corpus_format(raw_corpus, corpus_output)

    text2meta = build_title_index(corpus_output)

    missing_stats = convert_training_data_by_title(raw_train, train_output, text2meta)

    if sum(missing_stats.values()) > 0:
        print("⚠️ 以下 passages 在语料库中未找到匹配：")
        for k, v in missing_stats.items():
            print(f"  {k}: {v} 条")

if __name__ == "__main__":
    main()
