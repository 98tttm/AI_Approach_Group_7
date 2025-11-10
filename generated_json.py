import os
import json
import random

# === Cấu hình ===
OUTPUT_DIR = "generated_json"
NUM_FILES = 500
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Dữ liệu mẫu ===
first_names = ["Avery", "Taylor", "Jordan", "Riley", "Cameron", "Morgan", "Alex", "Peyton", "Casey", "Jamie"]
last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Davis", "Miller", "Wilson", "Moore", "Anderson"]
roles = ["Data Scientist", "Software Engineer", "Product Manager", "Business Analyst", "ML Engineer", "Backend Developer"]
skills_pool = ["Python", "C++", "SQL", "TensorFlow", "PyTorch", "Django", "React", "Kubernetes", "Pandas", "Scikit-learn", "AWS", "Docker", "Git"]
degrees = ["B.Tech", "B.Sc", "M.Tech", "M.Sc", "MBA", "Ph.D"]
fields = ["Computer Science", "Information Technology", "Data Science", "Business Analytics", "Artificial Intelligence"]
universities = ["MIT", "Stanford University", "Carnegie Mellon University", "IIT Delhi", "IIT Bombay", "Harvard University", "NUS", "University of Toronto"]
companies = ["Google", "Amazon", "Microsoft", "Meta", "VietInnovate", "FPT Software", "Tiki", "Shopee", "Grab", "VNPay"]

# === Hàm tạo ngẫu nhiên ===
def random_about():
    name = f"{random.choice(first_names)} {random.choice(last_names)}"
    role = random.choice(roles)
    return f"{name} {role}. Passionate about solving real-world problems with data and technology."

def random_skills():
    return ", ".join(random.sample(skills_pool, k=random.randint(5, 8))) + "."

def random_experience():
    exp = []
    for _ in range(random.randint(1, 3)):
        company = random.choice(companies)
        role = random.choice(roles)
        years = random.randint(1, 5)
        exp.append(f"Worked as a {role} at {company} for {years} year(s), focusing on data-driven solutions and scalable systems.")
    return " ".join(exp)

def random_education():
    degree = random.choice(degrees)
    field = random.choice(fields)
    uni = random.choice(universities)
    grad_year = random.randint(2015, 2024)
    return f"{degree} in {field} from {uni}, graduated {grad_year}."

# === Sinh file JSON ===
for i in range(1, NUM_FILES + 1):
    data = {
        "id": f"cv_{i:03d}.pdf",
        "sections": {
            "ABOUT": random_about(),
            "SKILLS": random_skills(),
            "WORK EXPERIENCE": random_experience(),
            "EDUCATION": random_education()
        }
    }

    out_path = os.path.join(OUTPUT_DIR, f"cv_{i:03d}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ Đã tạo xong {NUM_FILES} file JSON trong thư mục '{OUTPUT_DIR}'")
