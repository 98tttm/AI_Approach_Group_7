# -*- coding: utf-8 -*-
"""
Normalize fields in segmented CV sections
------------------------------------------
- Input: JSON files in Segmented_Text_v2/
- Output: Normalized_Text/ with cleaned and standardized fields
- Generate report normalization_report.txt

Run:
    python normalize_fields.py
"""

import os, re, json
from tqdm import tqdm
from collections import Counter

# ============ CONFIG ============
BASE = r"D:\Giai_Phap_AI\CVPDF_Parser"
INPUT_DIR  = os.path.join(BASE, "Segmented_Text_v2")
OUTPUT_DIR = os.path.join(BASE, "Normalized_Text")
LOG_PATH   = os.path.join(BASE, "logs", "normalization_report.txt")
DICT_PATH  = os.path.join(BASE, "normalization_dict.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ DICTIONARY ============
# Nếu bạn chưa có file normalization_dict.json, script sẽ tự sinh tạm
default_dict = {
    "b.tech": "bachelor of technology",
    "b.e.": "bachelor of engineering",
    "bsc": "bachelor of science",
    "bca": "bachelor of computer applications",
    "m.tech": "master of technology",
    "msc": "master of science",
    "mba": "master of business administration",
    "iit kanpur": "indian institute of technology kanpur",
    "iit bombay": "indian institute of technology bombay",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "py": "python",
    "tf": "tensorflow",
    "pytorch": "pytorch",
    "rto": "return to origin"
}

if os.path.exists(DICT_PATH):
    normalization_dict = json.load(open(DICT_PATH, "r", encoding="utf-8"))
else:
    normalization_dict = default_dict
    json.dump(default_dict, open(DICT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# ============ CLEANING FUNCTIONS ============
def basic_clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.,\-()&/:%]', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def normalize_with_dict(text):
    for k, v in normalization_dict.items():
        text = re.sub(rf'\b{k}\b', v, text)
    return text

def extract_years(text):
    years = re.findall(r"(19|20)\d{2}", text)
    return sorted(set(years))

def normalize_skills(text):
    # Tách kỹ năng bằng dấu , hoặc ;
    skills = re.split(r"[;,/\n]", text)
    skills = [basic_clean(s) for s in skills if len(s.strip()) > 1]
    skills = [normalize_with_dict(s) for s in skills]
    return sorted(set(skills))

def normalize_education(text):
    text = normalize_with_dict(basic_clean(text))
    degrees = re.findall(r"(bachelor|master|phd|doctorate|mba|b\.?tech|m\.?tech|bsc|msc)", text)
    schools = re.findall(r"(iit|university|college|institute|school)", text)
    years = extract_years(text)
    return {
        "raw": text.strip(),
        "degree": sorted(set(degrees)),
        "institution": sorted(set(schools)),
        "years": years
    }

def normalize_work(text):
    text = basic_clean(text)
    companies = re.findall(r"(pvt|ltd|inc|company|corp|technologies|solutions|lab[s]?)", text)
    titles = re.findall(r"(engineer|developer|scientist|analyst|manager|intern)", text)
    years = extract_years(text)
    return {
        "raw": text.strip(),
        "companies": sorted(set(companies)),
        "titles": sorted(set(titles)),
        "years": years
    }

def normalize_projects(text):
    text = normalize_with_dict(basic_clean(text))
    key_terms = re.findall(r"(project|system|model|web|app|application|object detection|analysis|predict)", text)
    return {
        "raw": text.strip(),
        "keywords": sorted(set(key_terms))
    }

# ============ MAIN ============
def main():
    report = Counter()
    for file in tqdm(os.listdir(INPUT_DIR), desc="Normalizing"):
        if not file.endswith(".json"):
            continue
        path = os.path.join(INPUT_DIR, file)
        data = json.load(open(path, "r", encoding="utf-8"))
        sections = data.get("sections", {})
        normalized = {}

        for key, val in sections.items():
            if not val.strip():
                continue

            if key == "SKILLS":
                normalized[key] = normalize_skills(val)
                report["skills"] += 1
            elif key == "EDUCATION":
                normalized[key] = normalize_education(val)
                report["education"] += 1
            elif key in ["WORK EXPERIENCE", "EXPERIENCE"]:
                normalized[key] = normalize_work(val)
                report["work"] += 1
            elif key == "PROJECTS":
                normalized[key] = normalize_projects(val)
                report["projects"] += 1
            else:
                # ABOUT, CERTIFICATIONS, ACTIVITIES, OTHER
                normalized[key] = basic_clean(val)
                report["others"] += 1

        out_data = {
            "id": data["id"],
            "normalized_sections": normalized
        }
        out_path = os.path.join(OUTPUT_DIR, file)
        json.dump(out_data, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # Summary report
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("=== NORMALIZATION REPORT ===\n")
        total = sum(report.values())
        for k, v in report.most_common():
            pct = (v / total * 100) if total else 0
            f.write(f"{k}: {v} ({pct:.1f}%)\n")
        f.write(f"\nTotal processed: {total}\n")
    print("✅ Done. Normalized files saved to:", OUTPUT_DIR)
    print("📄 Report:", LOG_PATH)

if __name__ == "__main__":
    main()
