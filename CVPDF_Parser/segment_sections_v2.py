# -*- coding: utf-8 -*-
"""
Segment CV sections (WORK EXPERIENCE / EDUCATION / SKILLS / PROJECTS / CERT / ABOUT / ACTIVITIES)
- Ưu tiên đọc layout thật từ PDF (PyMuPDF blocks) để giữ đúng thứ tự
- Heading scoring (lexicon + size + bold + ALLCAPS + độ dài)
- Gán nhãn dòng theo keyword score
- Viterbi smoothing để tránh nhảy nhãn lung tung
- Hậu xử lý (swap EDU/WORK, split PROJECTS/ACTIVITIES, gộp heading tương đương)
- Logger outliers (section rỗng, thứ tự lạ)

Chạy:
    python segment_sections_v2.py
"""

import os, re, json, math, sys
from collections import defaultdict, Counter
from tqdm import tqdm

try:
    import fitz  # PyMuPDF
    HAS_PDF = True
except Exception:
    HAS_PDF = False

# ====================== CONFIG ======================
BASE = r"D:\Giai_Phap_AI\CVPDF_Parser"
CLEAN_DIR = os.path.join(BASE, "Clean_Text")              # đầu vào text đã clean: {"id","text"}
PDF_DIR   = os.path.join(BASE, "ResumesPDF", "ResumesPDF")# nơi chứa file pdf gốc (nếu có)
OUT_DIR   = os.path.join(BASE, "Segmented_Text_v2")       # đầu ra
LOG_DIR   = os.path.join(BASE, "logs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Heading lexicon (mở rộng)
HEADINGS = [
    "WORK EXPERIENCE","EXPERIENCE","PROFESSIONAL EXPERIENCE","EMPLOYMENT",
    "EDUCATION","ACADEMICS","ACADEMIC BACKGROUND",
    "SKILL","SKILLS","TECHNICAL SKILLS",
    "PROJECT","PROJECTS",
    "CERTIFICATION","CERTIFICATIONS","LICENSES",
    "ACHIEVEMENT","ACHIEVEMENTS","AWARDS",
    "ACTIVITY","ACTIVITIES","EXTRA CURRICULAR",
    "ABOUT","PROFILE","SUMMARY","OBJECTIVE","PERSONAL SUMMARY"
]
HEADINGS_RE = re.compile(r"(?i)\b(" + "|".join([re.escape(h) for h in HEADINGS]) + r")\b")

# Nhãn chuẩn dùng cho Viterbi
LABELS = ["ABOUT","WORK EXPERIENCE","EDUCATION","SKILLS","PROJECTS","CERTIFICATIONS","ACHIEVEMENTS","ACTIVITIES","OTHER"]

# Transition ưu tiên (log-prob). Giữ đơn giản, thiên hướng thứ tự hợp lý:
TRANS = defaultdict(lambda: defaultdict(lambda: -2.5))  # default logP thấp
def set_trans(a, b, p):
    TRANS[a][b] = math.log(p)

for a in LABELS:
    TRANS[a][a] = math.log(0.60)  # stay
# plausible flows
flows = [
    ("ABOUT","WORK EXPERIENCE"),("ABOUT","EDUCATION"),("ABOUT","SKILLS"),
    ("WORK EXPERIENCE","PROJECTS"),("WORK EXPERIENCE","CERTIFICATIONS"),("WORK EXPERIENCE","ACHIEVEMENTS"),
    ("EDUCATION","SKILLS"),("EDUCATION","PROJECTS"),
    ("SKILLS","PROJECTS"),
    ("PROJECTS","ACHIEVEMENTS"),("PROJECTS","CERTIFICATIONS"),
    ("ACHIEVEMENTS","ACTIVITIES"),
    ("CERTIFICATIONS","ACHIEVEMENTS"),("CERTIFICATIONS","PROJECTS"),
]
for a,b in flows: set_trans(a,b,0.35)

# Keyword patterns cho emission score từng nhãn
P_EDU = re.compile(r"(?i)\b(B\.?Tech|M\.?Tech|Bachelor|Master|BSc|MSc|BE|ME|MBA|PhD|University|Institute|College|IIT|GPA|CGPA)\b")
P_WORK = re.compile(r"(?i)\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Present|Till Date|Engineer|Developer|Scientist|Analyst|Intern|Manager)\b")
P_SKILL = re.compile(r"(?i)\b(Python|Java|C\+\+|SQL|PySpark|TensorFlow|PyTorch|Keras|Docker|Kubernetes|NLP|Computer Vision|Machine Learning|Deep Learning|Git|Linux)\b")
P_PROJ = re.compile(r"(?i)\b(Project|Built|Implemented|Using|Developed|Object Detection|RTO|Web App|System|Module)\b")
P_CERT = re.compile(r"(?i)\b(Certification|Certified|License|Udemy|Coursera|AWS|Azure|GCP)\b")
P_ACHV = re.compile(r"(?i)\b(Award|Prize|Scholarship|Achievement|Ranked|Winner)\b")
P_ACT  = re.compile(r"(?i)\b(Activity|Club|Volunteer|Extracurricular)\b")
P_ABOUT= re.compile(r"(?i)\b(About|Summary|Objective|Profile)\b")

# Chuẩn hóa heading tương đương -> label chuẩn
CANON = {
    "EXPERIENCE":"WORK EXPERIENCE",
    "PROFESSIONAL EXPERIENCE":"WORK EXPERIENCE",
    "EMPLOYMENT":"WORK EXPERIENCE",
    "ACADEMICS":"EDUCATION",
    "ACADEMIC BACKGROUND":"EDUCATION",
    "TECHNICAL SKILLS":"SKILLS",
    "SKILL":"SKILLS",
    "PROJECT":"PROJECTS",
    "CERTIFICATION":"CERTIFICATIONS",
    "LICENSES":"CERTIFICATIONS",
    "ACHIEVEMENT":"ACHIEVEMENTS",
    "AWARDS":"ACHIEVEMENTS",
    "ACTIVITY":"ACTIVITIES",
    "PROFILE":"ABOUT",
    "SUMMARY":"ABOUT",
    "OBJECTIVE":"ABOUT",
    "PERSONAL SUMMARY":"ABOUT",
}

# =============== UTILITIES ===============
def rank_percentile(value, population):
    if not population: return 0.0
    sorted_vals = sorted(population)
    idx = sum(1 for v in sorted_vals if v <= value)
    return idx/len(sorted_vals)

def line_features(text):
    # nội dung thuần để gợi ý nhãn
    feats = {}
    feats["EDUCATION"] = 1.0 if P_EDU.search(text) else 0.0
    feats["WORK EXPERIENCE"] = 1.0 if P_WORK.search(text) else 0.0
    feats["SKILLS"] = 1.0 if P_SKILL.search(text) else 0.0
    feats["PROJECTS"] = 1.0 if P_PROJ.search(text) else 0.0
    feats["CERTIFICATIONS"] = 1.0 if P_CERT.search(text) else 0.0
    feats["ACHIEVEMENTS"] = 1.0 if P_ACHV.search(text) else 0.0
    feats["ACTIVITIES"] = 1.0 if P_ACT.search(text) else 0.0
    feats["ABOUT"] = 1.0 if P_ABOUT.search(text) else 0.0
    feats["OTHER"] = 0.0
    return feats

def heading_score(line_text, size_rank, is_bold):
    lex = 1 if HEADINGS_RE.search(line_text) else 0
    allcaps = 1 if line_text.isupper() and len(line_text) >= 3 else 0
    short = 1 if len(line_text) < 40 else 0
    score = 3*lex + 2*allcaps + 2*size_rank + 1*short + (1 if is_bold else 0)
    return score

def canonical_heading(h):
    h = h.upper().strip()
    return CANON.get(h, h)

def guess_label_by_heading(htext):
    m = HEADINGS_RE.search(htext)
    if not m: return None
    return canonical_heading(m.group(1))

# Viterbi cho chuỗi labels
def viterbi(emissions, start_label="ABOUT"):
    # emissions: list of dict label->logscore
    T = len(emissions)
    dp = [{l: (-1e9, None) for l in LABELS} for _ in range(T)]
    # init
    for l in LABELS:
        dp[0][l] = (emissions[0].get(l,-1e9) + TRANS[start_label][l], start_label)
    # iterate
    for t in range(1,T):
        for l in LABELS:
            best = (-1e9, None)
            for p in LABELS:
                val = dp[t-1][p][0] + TRANS[p][l] + emissions[t].get(l,-1e9)
                if val > best[0]:
                    best = (val, p)
            dp[t][l] = best
    # backtrack
    last_label = max(LABELS, key=lambda l: dp[T-1][l][0])
    seq = [last_label]
    for t in range(T-1,0,-1):
        seq.append(dp[t][seq[-1]][1])
    seq.reverse()
    return seq

# =============== EXTRACT LINES (PDF blocks or text fallback) ===============
def extract_lines_from_pdf(pdf_path):
    lines = []
    doc = fitz.open(pdf_path)
    for pno, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        sizes = []
        tmp = []
        for b in blocks:
            for l in b.get("lines", []):
                spans = l.get("spans", [])
                if not spans: continue
                txt = "".join(s["text"] for s in spans).strip()
                if not txt: continue
                size = max(s["size"] for s in spans)
                bold = any("Bold" in s["font"] for s in spans)
                y = l["bbox"][1]; x = l["bbox"][0]
                sizes.append(size)
                tmp.append({"text":txt, "x":x, "y":y, "size":size, "bold":bold, "page":pno})
        # percentile on page
        for d in tmp:
            d["size_rank"] = rank_percentile(d["size"], sizes)
            lines.append(d)
    # sort by (page, y, x)
    lines.sort(key=lambda d: (d["page"], d["y"], d["x"]))
    return lines

def extract_lines_from_text(txt):
    # fallback: chia theo dòng, ước lượng size_rank/bold
    raw_lines = [l.strip() for l in txt.split("\n") if l.strip()]
    lines = []
    for i,l in enumerate(raw_lines):
        allcaps = l.isupper()
        # heuristic size_rank: heading thường là ALLCAPS hoặc ngắn
        size_rank = 0.8 if allcaps or len(l) < 30 else 0.2
        lines.append({"text":l, "x":0, "y":i, "size":10+size_rank, "size_rank":size_rank, "bold":allcaps, "page":0})
    return lines

# ====================== MAIN ======================
def process_one(clean_json_path):
    data = json.load(open(clean_json_path, "r", encoding="utf-8"))
    file_id = data["id"]
    base_name = os.path.splitext(file_id)[0]
    pdf_path = os.path.join(PDF_DIR, base_name + ".pdf")

    # 1) Lines by layout if possible
    if HAS_PDF and os.path.exists(pdf_path):
        try:
            lines = extract_lines_from_pdf(pdf_path)
        except Exception:
            # fallback text
            lines = extract_lines_from_text(data["text"])
    else:
        lines = extract_lines_from_text(data["text"])

    if not lines:
        return {"id": file_id, "sections": {}}, {"empty": True}

    # 2) Detect headings (with score)
    headings = []

    for i, d in enumerate(lines):
        score = heading_score(d["text"], d.get("size_rank",0.0), d.get("bold",False))
        if score >= 5:  # ngưỡng heading
            label = guess_label_by_heading(d["text"])
            if label:
                headings.append((i, label))

    heading_idx = {h[0] for h in headings}  #lưu lại index các dòng heading
    # Merge headings too close
    merged = []
    for i,(idx,lbl) in enumerate(headings):
        if merged and idx - merged[-1][0] <= 1:
            # keep the stronger one by length heuristic
            if len(lines[idx]["text"]) < len(lines[merged[-1][0]]["text"]):
                continue
            merged[-1] = (idx, lbl)
        else:
            merged.append((idx,lbl))
    headings = merged

    # 3) Label each line by emission (content + proximity to headings)
    emissions = []
    near_label_hint = []
    for i, d in enumerate(lines):
        feats = line_features(d["text"])
        # proximity
        prox_bonus = {k:0.0 for k in LABELS}
        if headings:
            nearest = min(headings, key=lambda h: abs(h[0]-i))
            nh_lbl = nearest[1]
            near_label_hint.append(nh_lbl)
            prox_bonus[nh_lbl] = 0.7  # bonus gần heading
        else:
            near_label_hint.append(None)
        # emission score (log)
        emis = {}
        for lab in LABELS:
            base = feats.get(lab,0.0) + prox_bonus.get(lab,0.0)
            if base > 0: emis[lab] = math.log(0.5 + base)  # 0.5…1.5
        if not emis:
            emis = {"OTHER": math.log(0.6)}
        emissions.append(emis)

    # 4) Viterbi smoothing
    seq = viterbi(emissions, start_label="ABOUT")

    # 5) Build sections by grouping contiguous same label (skip OTHER)
    buckets = defaultdict(list)
    for i, (lab, d) in enumerate(zip(seq, lines)):
        if i in heading_idx:
            continue  #bỏ qua dòng heading để không dính vào nội dung
        lab = canonical_heading(lab)
        buckets[lab].append(d["text"])
    # Join
    sections = {}
    for lab, parts in buckets.items():
        if lab == "OTHER": continue
        content = "\n".join(parts).strip()
        if content:
            sections[lab] = content

    # 6) Post-fix (cứu sai)
    edu = sections.get("EDUCATION","")
    work = sections.get("WORK EXPERIENCE","")
    if (not work or len(work)<50) and re.search(P_WORK, edu):
        sections["WORK EXPERIENCE"], sections["EDUCATION"] = edu, work

    if "PROJECTS" in sections and re.search(r"(?i)\bACTIVITIES\b", sections["PROJECTS"]):
        left, right = re.split(r"(?i)\bACTIVITIES\b", sections["PROJECTS"], maxsplit=1)
        sections["PROJECTS"] = left.strip()
        if right.strip():
            sections["ACTIVITIES"] = right.strip()

    # 7) Outlier logging
    issues = []
    for key in ["WORK EXPERIENCE","EDUCATION","SKILLS"]:
        if key not in sections or len(sections[key].strip()) < 30:
            issues.append(f"{key}:EMPTY/SHORT")
    if not headings:
        issues.append("NO_HEADINGS_DETECTED")
    if issues:
        with open(os.path.join(LOG_DIR, "segment_outliers.txt"), "a", encoding="utf-8") as lf:
            lf.write(f"{file_id}\t" + ",".join(issues) + "\n")

    # 8) Clean redundant headings inside content
    for key, val in sections.items():
        cleaned = re.sub(r'(?i)\b(?:Work Experience|Education|Projects?|Skills?|Activities?|Certifications?)\b', '',
                         val)
        cleaned = re.sub(r'\n{2,}', '\n', cleaned).strip()
        sections[key] = cleaned

    return {"id": file_id, "sections": sections}, {"issues": issues}

def main():
    files = [f for f in os.listdir(CLEAN_DIR) if f.endswith(".json")]
    sum_issues = Counter()
    for f in tqdm(files, desc="Segmenting"):
        out, meta = process_one(os.path.join(CLEAN_DIR, f))
        with open(os.path.join(OUT_DIR, f), "w", encoding="utf-8") as fo:
            json.dump(out, fo, ensure_ascii=False, indent=2)
        for it in meta.get("issues",[]):
            sum_issues[it] += 1

    # summary
    with open(os.path.join(LOG_DIR, "segment_summary.txt"), "w", encoding="utf-8") as sf:
        sf.write("SEGMENT SUMMARY\n")
        for k,v in sum_issues.most_common():
            sf.write(f"{k}\t{v}\n")
    print("Done. Out:", OUT_DIR)
    print("Outliers log:", os.path.join(LOG_DIR, "segment_outliers.txt"))
    print("Summary   :", os.path.join(LOG_DIR, "segment_summary.txt"))

if __name__ == "__main__":
    main()
