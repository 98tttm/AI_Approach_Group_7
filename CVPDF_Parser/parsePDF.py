import os, fitz, json
from tqdm import tqdm

pdf_dir = r"D:\Giai_Phap_AI\CVPDF_Parser\ResumesPDF\ResumesPDF"
output_dir = r"D:\Giai_Phap_AI\CVPDF_Parser\Output_Text"
os.makedirs(output_dir, exist_ok=True)

for file in tqdm(os.listdir(pdf_dir)):
    if not file.endswith(".pdf"):
        continue
    path = os.path.join(pdf_dir, file)
    doc = fitz.open(path)

    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"

    # Lưu thành JSON (text-only)
    output_path = os.path.join(output_dir, file.replace(".pdf", ".json"))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"id": file, "text": text.strip()}, f, ensure_ascii=False)
