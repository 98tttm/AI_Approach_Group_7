import os

pdf_dir = r"..\CVPDF_Parser\ResumesPDF\ResumesPDF"
json_dir = r"..\CVPDF_Parser\ResumesJsonAnnotated\ResumesJsonAnnotated"

pdfs = sorted([f for f in os.listdir(pdf_dir) if f.endswith('.pdf')])
jsons = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])

#Đếm file
print("PDF count:", len(pdfs))
print("JSON count:", len(jsons))

# So sánh tên file
pdf_set = set(pdfs)
json_set = set(j.replace('_annotated.json', '.pdf') for j in jsons)
print("Missing:", pdf_set - json_set)
