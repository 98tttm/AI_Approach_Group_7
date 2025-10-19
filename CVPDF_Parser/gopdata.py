import os
import json

json_dir = r"D:\Giai_Phap_AI\CVPDF_Parser\ResumesJsonAnnotated\ResumesJsonAnnotated"
output_dir = r"D:\Giai_Phap_AI\data"
output_path = os.path.join(output_dir, "cv_ner_dataset.jsonl")

os.makedirs(output_dir, exist_ok=True)

def safe_json_dumps(data):
    return json.dumps(data, ensure_ascii=False).encode('utf-8', 'surrogatepass').decode('utf-8', 'ignore')

with open(output_path, "w", encoding="utf-8", errors="ignore") as out:
    for file in os.listdir(json_dir):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(json_dir, file), "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
            entry = {
                "id": file.replace("_annotated.json", ""),
                "text": data["text"],
                "entities": [
                    [a[0], a[1], a[2].split(":")[0]] for a in data["annotations"]
                ],
            }
            out.write(safe_json_dumps(entry) + "\n")

print(f"Saved merged dataset to: {output_path}")
