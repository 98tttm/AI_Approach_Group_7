import os, json, re
from tqdm import tqdm

input_dir = r"D:\Giai_Phap_AI\CVPDF_Parser\Clean_Text"
output_dir = r"D:\Giai_Phap_AI\CVPDF_Parser\Segmented_Text_v2"
os.makedirs(output_dir, exist_ok=True)

# Regex bắt các tiêu đề section (không phân biệt hoa thường)
section_pattern = re.compile(
    r'(?i)\b(WORK EXPERIENCE|EXPERIENCE|EDUCATION|ACADEMIC BACKGROUND|SKILLS?|TECHNICAL SKILLS|'
    r'PROJECTS?|CERTIFICATIONS?|ACHIEVEMENTS?|ABOUT|PROFILE|SUMMARY|ACTIVITIES?)\b'
)

for file in tqdm(os.listdir(input_dir)):
    if not file.endswith(".json"):
        continue

    path = os.path.join(input_dir, file)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    text = data["text"]

    # Làm sạch trước khi tìm section
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'(?i)(?<=\b)(WORKEXPERIENCE|PROJECTSACTIVITIES|EDUCATIONSKILLS)\b',
                  lambda m: re.sub(r'(?i)(WORK|EXPERIENCE|PROJECTS|ACTIVITIES|EDUCATION|SKILLS)',
                                   lambda n: n.group(0) + '\n', m.group(0)),
                  text)
    # đoạn trên tự thêm xuống dòng nếu heading dính liền nhau

    # Tìm vị trí section headings
    matches = list(section_pattern.finditer(text))
    sections = {}

    # Cắt theo thứ tự xuất hiện
    for i, match in enumerate(matches):
        section_name = match.group(1).upper().strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        sections[section_name] = content

    #Hậu xử lý logic nội dung (fix đảo thứ tự)
    edu = sections.get("EDUCATION", "")
    work = sections.get("WORK EXPERIENCE", "") or sections.get("EXPERIENCE", "")
    if len(work.strip()) < 50 and re.search(r'\b(Data|Engineer|Developer|Intern|Research|Analyst)\b', edu, re.I):
        sections["WORK EXPERIENCE"], sections["EDUCATION"] = edu, work

    #Xử lý PROJECTS bị dính ACTIVITIES
    if "PROJECTS" in sections and re.search(r'(?i)\bACTIVITIES\b', sections["PROJECTS"]):
        parts = re.split(r'(?i)\bACTIVITIES\b', sections["PROJECTS"], maxsplit=1)
        if len(parts) == 2:
            sections["PROJECTS"] = parts[0].strip()
            sections["ACTIVITIES"] = parts[1].strip()

    # Gộp các section tương đương
    if "TECHNICAL SKILLS" in sections:
        if "SKILLS" not in sections or len(sections["SKILLS"]) < len(sections["TECHNICAL SKILLS"]):
            sections["SKILLS"] = sections.pop("TECHNICAL SKILLS")

    #  Ghi ra file kết quả
    out_data = {"id": data["id"], "sections": sections}
    out_path = os.path.join(output_dir, file)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
