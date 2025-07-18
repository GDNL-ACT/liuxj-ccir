import json
from collections import defaultdict
import re

# 输入／输出文件路径
retrieval_path = "/home/liuxj25/LawLLM/CCIR/LexRAG/data/retrieval/res/retrieval_local_current_question.jsonl"
response_path  = "/home/liuxj25/LawLLM/CCIR/LexRAG/data/generation/generated_responses.jsonl"
output_path    = "merged.json"

# 1. 读取 retrieval.jsonl，构建 id -> 各轮 recall 列表 映射
retrieval_dict = {}
with open(retrieval_path, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        main_id = item['id']
        recalls = []
        for turn in item.get('conversation', []):
            # 每个 turn 中 question.recall
            recall = turn.get('question', {}).get('recall', [])
            recalls.append(recall)
        retrieval_dict[main_id] = recalls

# 2. 读取 response.jsonl，按照主 id 聚合多轮对话，并附带 per-turn recall
conversations = defaultdict(list)
id_pattern = re.compile(r"^(\d+)_turn(\d+)$")

with open(response_path, 'r', encoding='utf-8') as f:
    for line in f:
        resp = json.loads(line)
        raw_id = resp['id']
        m = id_pattern.match(raw_id)
        if m:
            main_id, turn_idx = m.group(1), int(m.group(2))
        else:
            print("Error")
        # 获取对应轮次的 recall
        recall_list = retrieval_dict.get(main_id, [])
        # turn 索引从 1 开始，对应列表索引 0
        recall = recall_list[turn_idx-1] if 0 < turn_idx <= len(recall_list) else []

        entry = {
            "question": resp.get("question", ""),
            "recall": recall,
            "response": resp.get("response", "")
        }
        conversations[main_id].append((turn_idx, entry))

# 3. 按轮次排序，并生成最终列表
merged = []
for main_id, turns in conversations.items():
    sorted_entries = [entry for _, entry in sorted(turns, key=lambda x: x[0])]
    merged.append({
        "id": main_id,
        "conversation": sorted_entries
    })

# 按 id 升序
merged.sort(key=lambda x: int(x['id']))

# 4. 写入 merged.json
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print(f"合并完成，结果已写入 {output_path}，共包含 {len(merged)} 个对话。")
