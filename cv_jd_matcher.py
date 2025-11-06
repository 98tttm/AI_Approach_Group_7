"""
FILE: cv_jd_matcher.py
PURPOSE: Rule-Based CV-JD Scoring System
=============================================================================
HƯỚNG DẪN SỬ DỤNG:
=============================================================================

FILE NÀY LÀM GÌ?
- Chấm điểm CV theo RULES CỨNG (không dùng AI)
- So sánh CV với Job Description (JD)
- Tính điểm từ 0-100 dựa trên 4 tiêu chí

CÁCH DÙNG:
1. Import class này: from cv_jd_matcher import SimpleCVJDMatcher
2. Tạo object: matcher = SimpleCVJDMatcher()
3. Chấm điểm: result = matcher.score_cv_against_jd(cv_data, jd_data)

INPUT:
- cv_data: Dictionary với format từ Task 2 (NER output)
- jd_data: Dictionary với job requirements

OUTPUT:
- Dictionary với total_score và breakdown chi tiết

VÍ DỤ:
    matcher = SimpleCVJDMatcher()
    score = matcher.score_cv_against_jd(my_cv, my_jd)
    print(f"Total Score: {score['total_score']}/100")

=============================================================================
"""

import re
import json
from typing import Dict, List


class SimpleCVJDMatcher:
    """
    Class chấm điểm CV dựa trên Job Description

    Scoring Weights:
    - Skills: 35 điểm (35%)
    - Work Experience: 40 điểm (40%)
    - Education: 15 điểm (15%)
    - About/Summary: 10 điểm (10%)
    Total: 100 điểm
    """

    def __init__(self):
        """
        Khởi tạo matcher

        KHÔNG CẦN TRUYỀN GÌ VÀO - chỉ cần:
        matcher = SimpleCVJDMatcher()
        """
        # Degree levels để so sánh học vấn
        self.degree_levels = {
            "phd": 4, "doctorate": 4,
            "master": 3, "mba": 3, "m.s.": 3, "m.a.": 3, "m.sc": 3,
            "bachelor": 2, "b.s.": 2, "b.a.": 2, "b.e.": 2, "b.sc": 2, "b.tech": 2,
            "associate": 1, "diploma": 1
        }

        # --- Normalization maps & aliases (do NOT change weights) ---
        # Simple stemming/lemmatization map for verbs/plurals frequently used in CVs/JDs
        self._stem_map = {
            "built": "build",
            "building": "build",
            "led": "lead",
            "leads": "lead",
            "apis": "api",
            "services": "service",
            "queries": "query",
            "optimizing": "optimize",
            "optimized": "optimize",
            "deployments": "deployment",
            "databases": "database"
        }

        # Skill aliases -> canonical
        self._skill_alias = {
            "postgres": "postgresql",
            "postgre": "postgresql",
            "k8s": "kubernetes",
            "gke": "kubernetes",
            "eks": "kubernetes",
            "rest": "rest api",
            "restful": "rest api",
            "ci/cd": "cicd",
            "gitlab ci": "cicd",
            "mq": "rabbitmq",
            "elastic search": "elasticsearch"
        }

        # Child tokens that imply a parent skill
        self._implies_parent = {
            "aws": {"ec2", "s3", "rds", "lambda", "ecs", "eks", "cloudwatch"},
            "rabbitmq": {"amqp"},
        }

    def score_cv_against_jd(self, cv_data: Dict, jd_data: Dict) -> Dict:
        """
        HÀM CHÍNH - Chấm điểm CV với JD

        CÁCH DÙNG:
            result = matcher.score_cv_against_jd(cv_data, jd_data)
            print(result['total_score'])  # In ra tổng điểm

        INPUT:
            cv_data: Dictionary chứa CV info
                Format: {
                    "id": "cv_1.pdf",
                    "sections": {
                        "ABOUT": "...",
                        "SKILLS": "...",
                        "WORK EXPERIENCE": "...",
                        "EDUCATION": "..."
                    }
                }

            jd_data: Dictionary chứa JD requirements
                Format: {
                    "job_id": "JD_001",
                    "title": "Senior Backend Engineer",
                    "requirements": {
                        "skills": ["Python", "Java", ...],
                        "min_years_experience": 5,
                        "key_responsibilities": [...],
                        "min_education": "Bachelor"
                    }
                }

        OUTPUT:
            Dictionary: {
                "total_score": 72.5,
                "skills_score": 23.5,
                "experience_score": 32.0,
                "education_score": 12.0,
                "about_score": 5.0,
                "rating": "Good",
                "breakdown": {...}  # Chi tiết từng phần
            }
        """

        sections = cv_data.get('sections', {})
        requirements = jd_data.get('requirements', {})

        # 1. SKILLS MATCHING (35 điểm)
        skills_result = self._score_skills(
            sections.get('SKILLS', ''),
            requirements.get('skills', [])
        )

        # 2. WORK EXPERIENCE (40 điểm)
        experience_result = self._score_experience(
            sections.get('WORK EXPERIENCE', ''),
            requirements.get('min_years_experience', 0),
            requirements.get('key_responsibilities', [])
        )

        # 3. EDUCATION (15 điểm)
        education_result = self._score_education(
            sections.get('EDUCATION', ''),
            requirements.get('min_education', 'Bachelor')
        )

        # 4. ABOUT/SUMMARY (10 điểm)
        about_result = self._score_about(
            sections.get('ABOUT', ''),
            jd_data
        )

        # TỔNG HỢP
        total_score = (
                skills_result['score'] +
                experience_result['score'] +
                education_result['score'] +
                about_result['score']
        )

        return {
            'cv_id': cv_data.get('id', 'Unknown'),
            'jd_id': jd_data.get('job_id', 'Unknown'),
            'jd_title': jd_data.get('title', 'Unknown'),
            'total_score': round(total_score, 2),
            'skills_score': skills_result['score'],
            'experience_score': experience_result['score'],
            'education_score': education_result['score'],
            'about_score': about_result['score'],
            'rating': self._get_rating(total_score),
            'breakdown': {
                'skills': skills_result,
                'experience': experience_result,
                'education': education_result,
                'about': about_result
            }
        }

    def _normalize_phrase(self, s: str) -> str:
        """
        Normalize a free-text phrase for robust matching:
        - lowercase
        - replace separators (/, -, _) with space
        - collapse whitespace
        - apply simple aliasing (skill aliases)
        """
        if not s:
            return ""
        x = s.lower()
        # unify separators
        x = x.replace("/", " ").replace("-", " ").replace("_", " ")
        # drop trailing punctuation commonly found in skills lists
        x = re.sub(r"[.,;:]+", " ", x)
        # collapse whitespace
        x = re.sub(r"\s+", " ", x).strip()

        # apply aliases (whole-phrase)
        if x in self._skill_alias:
            x = self._skill_alias[x]
        return x

    def _normalize_token(self, w: str) -> str:
        """
        Normalize a single token (used for responsibilities keywords).
        """
        if not w:
            return ""
        w = re.sub(r"[^a-z0-9+]", "", w.lower())
        if w in self._stem_map:
            return self._stem_map[w]
        if w in self._skill_alias:
            return self._skill_alias[w]
        return w

    def _expand_parent_skills(self, phrases: List[str]) -> List[str]:
        """
        If a phrase implies a parent skill (e.g., 'ec2' implies 'aws'),
        add the parent token so JD 'aws' can match.
        """
        out = set(phrases)
        for parent, children in self._implies_parent.items():
            if any(child in out for child in children):
                out.add(parent)
        return list(out)

    def _expand_alternatives(self, s: str) -> List[str]:
        """
        Expand alternative forms inside a single JD skill entry.
        Examples:
          "Django or FastAPI" -> ["django", "fastapi"]
          "MySQL/PostgreSQL"  -> ["mysql", "postgresql"]
          "Java | Kotlin"     -> ["java", "kotlin"]
        """
        if not s:
            return []
        x = self._normalize_phrase(s)
        # split on common OR separators: " or ", "/", "|"
        parts = re.split(r"\s+or\s+|/|\|", x)
        alts = [p.strip() for p in parts if p and p.strip()]
        # de-duplicate
        dedup = []
        for a in alts:
            if a not in dedup:
                dedup.append(a)
        return dedup if dedup else [x]

    def _score_skills(self, cv_skills_text: str, jd_skills: List[str]) -> Dict:
        """
        Chấm điểm SKILLS (35 điểm)
        - Giữ nguyên trọng số 35
        - Cải thiện parsing: ưu tiên tách theo dấu phẩy/; và xuống dòng để giữ nguyên cụm
        - Chuẩn hoá cụm & alias
        - Partial match: one contains other (0.8)
        - Thêm suy diễn cha-con (EC2/S3 ⇒ AWS)
        """
        if not cv_skills_text or not jd_skills:
            return {
                'score': 0.0,
                'matched_count': 0,
                'total_required': len(jd_skills),
                'match_percentage': 0.0,
                'details': 'No skills data'
            }

        # 1) Parse CV skills by phrase (keep phrases like "spring boot", "rest api")
        # split by comma/semicolon/pipe/newline first
        raw_parts = re.split(r"[,\n;|]+", cv_skills_text)
        cv_phrases = [self._normalize_phrase(p) for p in raw_parts if p and p.strip()]
        # fallback: if nothing parsed (all spaces), split by space
        if not cv_phrases:
            cv_phrases = [self._normalize_phrase(p) for p in cv_skills_text.split() if p.strip()]

        # 2) Expand parent skills (ec2/s3/… ⇒ aws)
        cv_phrases = self._expand_parent_skills(cv_phrases)

        matched_count = 0.0
        match_details = []

        # Pre-normalize JD skills, expanding alternatives
        jd_norm = []
        for s in jd_skills:
            alts = self._expand_alternatives(s)
            jd_norm.append((s, alts))

        for original_jd_skill, alt_list in jd_norm:
            best = 0.0
            best_with = None
            best_alt = None

            for alt in alt_list:
                for cvp in cv_phrases:
                    if not cvp:
                        continue

                    if cvp == alt:
                        score = 1.0
                    elif alt in cvp or cvp in alt:
                        score = 0.8
                    else:
                        score = 0.0

                    if score > best:
                        best = score
                        best_with = cvp
                        best_alt = alt

                # short-circuit if exact already found
                if best == 1.0:
                    break

            matched_count += best
            match_details.append({
                'jd_skill': original_jd_skill,
                'expanded_alternative_used': best_alt if best > 0 else None,
                'matched_with': best_with if best > 0 else None,
                'match_score': best
            })

        match_percentage = matched_count / len(jd_skills) if jd_skills else 0.0
        final_score = match_percentage * 35

        return {
            'score': round(final_score, 2),
            'matched_count': round(matched_count, 2),
            'total_required': len(jd_skills),
            'match_percentage': round(match_percentage * 100, 1),
            'details': match_details
        }

    def _score_experience(self, cv_experience: str, jd_min_years: int,
                          jd_responsibilities: List[str]) -> Dict:
        """
        Chấm điểm WORK EXPERIENCE (40 điểm)

        GỒM 2 PHẦN:
        A. Years Match (20 điểm) - So sánh số năm kinh nghiệm
        B. Responsibilities Match (20 điểm) - So sánh công việc đã làm
        """

        if not cv_experience:
            return {
                'score': 0.0,
                'years_score': 0.0,
                'responsibilities_score': 0.0,
                'details': 'No experience data'
            }

        # PART A: Years Match (20 điểm)
        years_result = self._extract_and_score_years(cv_experience, jd_min_years)

        # PART B: Responsibilities Match (20 điểm)
        resp_result = self._score_responsibilities(cv_experience, jd_responsibilities)

        total_exp_score = years_result['score'] + resp_result['score']

        return {
            'score': round(total_exp_score, 2),
            'years_score': years_result['score'],
            'responsibilities_score': resp_result['score'],
            'years_details': years_result,
            'responsibilities_details': resp_result
        }

    def _extract_and_score_years(self, experience_text: str, jd_min_years: int) -> Dict:
        """
        Trích xuất và chấm điểm SỐ NĂM KINH NGHIỆM (20 điểm)

        EXTRACTION PATTERNS:
        - "5 year(s)" → 5 years
        - "from 2019 to 2023" → 4 years

        SCORING:
        - CV ≥ JD × 1.5: 20 điểm (Excellent)
        - CV ≥ JD: 18 điểm (Good)
        - CV ≥ JD × 0.8: 14 điểm (Acceptable)
        - CV < JD × 0.8: Proportional (0-13 điểm)
        """

        # Pattern 1: "X year(s)"
        year_pattern = r'(\d+)\s*year\(?s?\)?'
        year_matches = re.findall(year_pattern, experience_text.lower())
        total_years_v1 = sum(int(y) for y in year_matches) if year_matches else 0

        # Pattern 2: "from YYYY to YYYY"
        date_pattern = r'from\s+(\d{4})\s+to\s+(\d{4})'
        date_matches = re.findall(date_pattern, experience_text)

        if date_matches:
            # Take the longest range
            years_from_dates = [int(end) - int(start) for start, end in date_matches]
            total_years_v2 = max(years_from_dates)
        else:
            total_years_v2 = 0

        # Take maximum (more conservative)
        cv_years = max(total_years_v1, total_years_v2)

        # Scoring logic
        if jd_min_years == 0:
            score = 20.0  # No requirement
        elif cv_years >= jd_min_years * 1.5:
            score = 20.0  # Excellent
        elif cv_years >= jd_min_years:
            score = 18.0  # Good
        elif cv_years >= jd_min_years * 0.8:
            score = 14.0  # Acceptable
        else:
            # Proportional for below 80%
            ratio = cv_years / (jd_min_years * 0.8) if jd_min_years > 0 else 0
            score = ratio * 14.0

        return {
            'score': round(score, 2),
            'cv_years': cv_years,
            'jd_min_years': jd_min_years,
            'ratio': round(cv_years / jd_min_years, 2) if jd_min_years > 0 else 0,
            'level': self._get_years_level(cv_years, jd_min_years)
        }

    def _get_years_level(self, cv_years: int, jd_min_years: int) -> str:
        """Helper: Xác định level của years"""
        if jd_min_years == 0:
            return 'No requirement'
        elif cv_years >= jd_min_years * 1.5:
            return 'Excellent'
        elif cv_years >= jd_min_years:
            return 'Good'
        elif cv_years >= jd_min_years * 0.8:
            return 'Acceptable'
        else:
            return 'Below requirement'

    def _score_responsibilities(self, cv_experience: str,
                                jd_responsibilities: List[str]) -> Dict:
        """
        Chấm điểm RESPONSIBILITIES MATCH (20 điểm)

        LOGIC:
        - Với mỗi JD responsibility:
          1. Extract keywords (bỏ stop words)
          2. Check keywords có trong CV không
          3. Match ratio ≥ 50% → considered matched

        VD:
        JD: "Design and build RESTful APIs"
        Keywords: [design, build, restful, apis]
        CV: "Built RESTful APIs for platform"
        Found: [built, restful, apis] → 3/4 = 75% → Matched!
        """

        if not jd_responsibilities:
            return {
                'score': 20.0,  # No requirements
                'matched_count': 0,
                'total_required': 0,
                'match_percentage': 100.0
            }

        cv_exp_lower = self._normalize_phrase(cv_experience)

        matched_count = 0.0
        details = []

        for responsibility in jd_responsibilities:
            # Extract keywords
            keywords = self._extract_keywords(responsibility)

            # Check keywords in CV
            found_keywords = [kw for kw in keywords if kw in cv_exp_lower]

            # Calculate match ratio
            match_ratio = len(found_keywords) / len(keywords) if keywords else 0

            # Consider matched if ≥ 40% keywords found (softer threshold)
            if match_ratio >= 0.4:
                matched_count += match_ratio  # Partial credit
                matched = True
            else:
                matched = False

            details.append({
                'responsibility': responsibility,
                'keywords': keywords,
                'found_keywords': found_keywords,
                'match_ratio': round(match_ratio, 2),
                'matched': matched
            })

        # Calculate final score
        match_percentage = matched_count / len(jd_responsibilities)
        final_score = match_percentage * 20

        return {
            'score': round(final_score, 2),
            'matched_count': round(matched_count, 2),
            'total_required': len(jd_responsibilities),
            'match_percentage': round(match_percentage * 100, 1),
            'details': details
        }

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Trích xuất keywords từ text
        - Loại stop words
        - Chỉ giữ từ dài > 3
        - Chuẩn hoá token (stemming/alias)
        """
        stop_words = {
            'and', 'the', 'for', 'with', 'a', 'an', 'in', 'on', 'at',
            'to', 'of', 'from', 'by', 'as', 'is', 'was', 'are', 'been'
        }
        words = re.split(r"\s+", text.lower())
        keywords = []
        for w in words:
            w = re.sub(r"[^\w+/.-]", "", w)  # keep + for c++, / for ci/cd, . for versions
            if len(w) <= 3:
                continue
            if w in stop_words:
                continue
            kw = self._normalize_token(w)
            if kw and len(kw) > 3 and kw not in stop_words:
                keywords.append(kw)
        return keywords

    def _score_education(self, cv_education: str, jd_min_education: str) -> Dict:
        """
        Chấm điểm EDUCATION (15 điểm)

        LOGIC:
        - Extract degree level từ CV
        - Compare với JD requirement
        - Simple scoring:
          * CV ≥ JD: 15 điểm
          * CV = JD - 1: 10 điểm
          * CV < JD - 1: 5 điểm

        DEGREE LEVELS:
        PhD = 4, Master = 3, Bachelor = 2, Associate = 1
        """

        if not cv_education:
            return {
                'score': 0.0,
                'cv_degree': 'Unknown',
                'cv_level': 0,
                'jd_level': 0,
                'meets_requirement': False
            }

        # Extract CV degree level
        cv_edu_lower = cv_education.lower()
        cv_level = 0
        cv_degree = 'Unknown'

        for degree_name, level in sorted(self.degree_levels.items(),
                                         key=lambda x: -x[1]):
            if degree_name in cv_edu_lower:
                cv_level = level
                cv_degree = degree_name.upper()
                break

        # JD required level
        jd_level = self.degree_levels.get(jd_min_education.lower(), 2)

        # Scoring
        if cv_level >= jd_level:
            score = 15.0  # Meets or exceeds
        elif cv_level == jd_level - 1:
            score = 10.0  # One level below
        elif cv_level > 0:
            score = 5.0  # Has degree but below
        else:
            score = 0.0  # No degree

        return {
            'score': score,
            'cv_degree': cv_degree,
            'cv_level': cv_level,
            'jd_level': jd_level,
            'jd_required': jd_min_education,
            'meets_requirement': cv_level >= jd_level
        }

    def _score_about(self, cv_about: str, jd_data: Dict) -> Dict:
        """
        Chấm điểm ABOUT/SUMMARY (10 điểm)

        GỒM 2 PHẦN:
        A. Title Match (5 điểm) - JD title có trong About không?
        B. Keyword Mentions (5 điểm) - Top skills có được mention không?
        """

        if not cv_about:
            return {
                'score': 0.0,
                'title_score': 0.0,
                'keyword_score': 0.0
            }

        about_lower = cv_about.lower()

        # PART A: Title Match (5 điểm)
        jd_title_words = jd_data.get('title', '').lower().split()
        title_matches = 0

        for word in jd_title_words:
            if len(word) > 3 and word in about_lower:
                title_matches += 1

        # Normalize to 0-5
        title_score = min(title_matches * 1.5, 5.0)

        # PART B: Keyword Mentions (5 điểm)
        # Use top 5 skills from JD, expanding alternatives for about matching
        jd_skills = jd_data.get('requirements', {}).get('skills', [])
        top_raw = jd_skills[:5]
        # expand alternatives for about matching
        expanded_sets = [self._expand_alternatives(s) for s in top_raw]

        keyword_count = 0
        for alts in expanded_sets:
            # if any alternative appears in ABOUT, count 1
            if any(alt in about_lower for alt in alts):
                keyword_count += 1
        keyword_score = min(keyword_count, 5.0)

        return {
            'score': round(title_score + keyword_score, 2),
            'title_score': round(title_score, 2),
            'keyword_score': keyword_score,
            'title_words_matched': title_matches,
            'keywords_mentioned': keyword_count
        }

    def _get_rating(self, score: float) -> str:
        """
        Chuyển điểm số thành rating

        THRESHOLDS:
        - 85-100: Excellent
        - 70-84: Very Good
        - 60-69: Good
        - 50-59: Fair
        - 40-49: Below Average
        - 0-39: Poor
        """
        if score >= 85:
            return 'Excellent'
        elif score >= 70:
            return 'Very Good'
        elif score >= 60:
            return 'Good'
        elif score >= 50:
            return 'Fair'
        elif score >= 40:
            return 'Below Average'
        else:
            return 'Poor'


# =============================================================================
# EXAMPLE USAGE - CÁCH DÙNG MẪU
# =============================================================================

if __name__ == "__main__":
    """
    Đây là ví dụ cách sử dụng SimpleCVJDMatcher

    CHẠY FILE NÀY ĐỂ TEST:
    python cv_jd_matcher.py
    """

    # Example CV data (từ Task 2 NER)
    example_cv = {
        "id": "cv_example.pdf",
        "sections": {
            "ABOUT": "Senior Software Engineer with 6 years experience in Python and AWS. Specialized in building scalable backend services.",
            "SKILLS": "Python, Django, PostgreSQL, AWS, Docker, REST API, Git",
            "WORK EXPERIENCE": "Built RESTful APIs for e-commerce platform; 3 year(s) of impact. Optimized database queries; 2 year(s). Led backend team; 1 year(s). Worked at TechCorp from 2018 to 2024.",
            "EDUCATION": "B.S. in Computer Science from State University, graduated 2018."
        }
    }

    # Example JD data
    example_jd = {
        "job_id": "JD_001",
        "title": "Senior Backend Engineer",
        "company": "TechViet",
        "requirements": {
            "skills": ["Python", "Django", "PostgreSQL", "AWS", "Docker"],
            "min_years_experience": 5,
            "key_responsibilities": [
                "Design and build RESTful APIs",
                "Optimize database queries",
                "Lead technical teams"
            ],
            "min_education": "Bachelor"
        }
    }

    # Tạo matcher
    print("=" * 60)
    print("DEMO: CV-JD MATCHING")
    print("=" * 60)

    matcher = SimpleCVJDMatcher()

    # Chấm điểm
    result = matcher.score_cv_against_jd(example_cv, example_jd)

    # In kết quả
    print(f"\nCV: {result['cv_id']}")
    print(f"JD: {result['jd_title']}")
    print(f"\n{'=' * 60}")
    print(f"TOTAL SCORE: {result['total_score']}/100")
    print(f"RATING: {result['rating']}")
    print(f"{'=' * 60}")

    print(f"\nBREAKDOWN:")
    print(f"  Skills:       {result['skills_score']:.2f}/35")
    print(f"  Experience:   {result['experience_score']:.2f}/40")
    print(f"  Education:    {result['education_score']:.2f}/15")
    print(f"  About:        {result['about_score']:.2f}/10")

    print(f"\nDETAILS:")
    print(f"  Skills Match: {result['breakdown']['skills']['match_percentage']}%")
    print(f"  Years: {result['breakdown']['experience']['years_details']['cv_years']} years")
    print(f"  Education: {result['breakdown']['education']['cv_degree']}")

    print("\n" + "=" * 60)
    print("TEST COMPLETED!")
    print("=" * 60)
