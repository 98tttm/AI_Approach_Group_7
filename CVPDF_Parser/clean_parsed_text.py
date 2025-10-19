import os, json, re
from tqdm import tqdm

input_dir = r"D:\Giai_Phap_AI\CVPDF_Parser\Output_Text"
output_dir = r"D:\Giai_Phap_AI\CVPDF_Parser\Clean_Text"
os.makedirs(output_dir, exist_ok=True)

def clean_text(text):
    text = re.sub(r'[ \t]+', ' ', text)           # bỏ tab và nhiều space
    text = re.sub(r'\n{2,}', '\n', text)          # giảm nhiều dòng trống thành 1
    text = re.sub(r'[^ -~\n]', '', text)          # loại ký tự không in được
    text = text.strip()
    return text

for file in tqdm(os.listdir(input_dir)):
    if not file.endswith(".json"): continue
    path = os.path.join(input_dir, file)
    data = json.load(open(path, "r", encoding="utf-8", errors="ignore"))
    text = clean_text(data["text"])
    json.dump({"id": data["id"], "text": text}, open(os.path.join(output_dir, file), "w", encoding="utf-8"), ensure_ascii=False)
