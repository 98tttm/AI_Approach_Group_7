import json
from collections import Counter

path = r"D:\Giai_Phap_AI\data\cv_ner_dataset.jsonl"

# Đếm số CV (số dòng)
count = 0
with open(path, "r", encoding="utf-8") as f:
    for _ in f:
        count += 1
print(f"Tổng số CV trong dataset: {count:,}")

label_counter = Counter()
total_len = 0
num_docs = 0

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        num_docs += 1
        total_len += len(data["text"])
        for e in data["entities"]:
            label_counter[e[2]] += 1

print(f"Tổng CV: {num_docs}")
print(f"Độ dài text trung bình: {total_len/num_docs:.0f} ký tự")
print(f"Thống kê nhãn (entity types):")
for label, count in label_counter.most_common():
    print(f"   {label:<10} → {count:,}")