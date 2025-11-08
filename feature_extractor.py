"""
FILE: feature_extractor.py
PURPOSE: Extract features từ CV+JD để train ML model

=============================================================================
HƯỚNG DẪN SỬ DỤNG:
=============================================================================

FILE NÀY LÀM GÌ?
- Chuyển CV + JD thành NUMBERS (features) để ML model hiểu được
- ML model chỉ hiểu numbers, không hiểu text
- Extract 40+ features từ mỗi CV-JD pair

TẠI SAO CẦN FILE NÀY?
- Rule-based cho ra 1 điểm (VD: 68)
- Nhưng ML model cần nhiều thông tin hơn để học
- File này tạo ra 40 con số mô tả CV+JD

CÁCH DÙNG:
1. Import: from feature_extractor import FeatureExtractor
2. Tạo object: extractor = FeatureExtractor()
3. Extract: features = extractor.extract_features(cv_data, jd_data)

OUTPUT:
- List of 40 numbers
- VD: [0.68, 5, 10, 3, 1.2, 0.75, ...]

VÍ DỤ:
    extractor = FeatureExtractor()
    features = extractor.extract_features(my_cv, my_jd)
    print(f"Extracted {len(features)} features")

=============================================================================
"""

import re
from typing import Dict, List


class FeatureExtractor:
    """
    Class trích xuất features từ CV + JD

    Output: 40+ numerical features cho ML model

    CATEGORIES:
    1. Skills Features (12 features)
    2. Experience Features (15 features)
    3. Education Features (6 features)
    4. Text Quality Features (7 features)
    """

    def __init__(self):
        """
        Khởi tạo extractor

        KHÔNG CẦN TRUYỀN GÌ:
        extractor = FeatureExtractor()
        """
        self.feature_names = []  # Sẽ được set sau khi extract

        # --- Normalization maps & aliases (align with matcher; do NOT change weights) ---
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
            "elastic search": "elasticsearch",
            "fast api": "fastapi"
        }

        # Child tokens that imply a parent skill
        self._implies_parent = {
            "aws": {"ec2", "s3", "rds", "lambda", "ecs", "eks", "cloudwatch"},
            "rabbitmq": {"amqp"},
        }

    def extract_features(self, cv_data: Dict, jd_data: Dict) -> List[float]:
        """
        HÀM CHÍNH - Extract features từ CV + JD

        CÁCH DÙNG:
            features = extractor.extract_features(cv_data, jd_data)
            # features = [0.6, 5, 10, 3, ...]  (40 numbers)

        INPUT:
            cv_data: CV dictionary (từ Task 2)
            jd_data: JD dictionary

        OUTPUT:
            List of 40 floats - numerical features

        VÍ DỤ OUTPUT:
            [
                0.68,    # skills_match_percentage
                5,       # num_skills_matched
                10,      # num_skills_in_cv
                ...      # 37 features nữa
            ]
        """

        sections = cv_data.get('sections', {})
        requirements = jd_data.get('requirements', {})

        # Dictionary để lưu features
        features_dict = {}

        # 1. Skills Features (12 features)
        skills_features = self._extract_skills_features(
            sections.get('SKILLS', ''),
            requirements.get('skills', [])
        )
        features_dict.update(skills_features)

        # 2. Experience Features (15 features)
        exp_features = self._extract_experience_features(
            sections.get('WORK EXPERIENCE', ''),
            requirements.get('min_years_experience', 0),
            requirements.get('key_responsibilities', [])
        )
        features_dict.update(exp_features)

        # 3. Education Features (6 features)
        edu_features = self._extract_education_features(
            sections.get('EDUCATION', ''),
            requirements.get('min_education', 'Bachelor')
        )
        features_dict.update(edu_features)

        # 4. Text Quality Features (7 features)
        text_features = self._extract_text_features(sections)
        features_dict.update(text_features)

        # Convert dictionary to list (order matters!)
        self.feature_names = list(features_dict.keys())
        feature_values = list(features_dict.values())

        return feature_values

    def _extract_skills_features(self, cv_skills_text: str,
                                 jd_skills: List[str]) -> Dict:
        """
        Extract SKILLS FEATURES (expanded & normalized)
        - Keep phrases (don't break "spring boot", "rest api")
        - Understand OR in JD skills ("Django or FastAPI")
        - Aliases (k8s->kubernetes, postgres->postgresql, restful->rest api)
        - Add core boolean flags for BE Python track
        """
        features = {}

        # 0) Parse CV skills by phrase first (comma/semicolon/pipe/newline)
        if cv_skills_text:
            raw_parts = re.split(r"[,\n;|]+", cv_skills_text)
            cv_phrases = [self._normalize_phrase(p) for p in raw_parts if p and p.strip()]
            if not cv_phrases:
                cv_phrases = [self._normalize_phrase(p) for p in cv_skills_text.split() if p.strip()]
        else:
            cv_phrases = []

        # Expand parent skills (ec2/s3/rds/lambda => aws)
        cv_phrases = self._expand_parent_skills(cv_phrases)

        # 1) Matching calculation (consistent with matcher)
        matched_count = 0.0
        jd_norm = []
        for s in jd_skills or []:
            alts = self._expand_alternatives(s)
            jd_norm.append((s, alts))

        match_details = []
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

                if best == 1.0:
                    break

            matched_count += best
            match_details.append({
                'jd_skill': original_jd_skill,
                'expanded_alternative_used': best_alt if best > 0 else None,
                'matched_with': best_with if best > 0 else None,
                'match_score': best
            })

        match_percentage = (matched_count / len(jd_skills)) if jd_skills else 0.0

        # Feature 1-4: Basic matching metrics (aligned with matcher semantics)
        features['skills_match_percentage'] = round(match_percentage, 4)
        features['num_skills_matched'] = int(round(matched_count))
        features['num_skills_in_cv'] = len(cv_phrases)
        features['num_skills_in_jd'] = len(jd_skills or [])

        # Strings for convenience
        cv_skills_str = " ".join(cv_phrases)

        # 2) Category indicators (binary) — add FastAPI and separate cloud flags
        prog_langs = ['python', 'java', 'javascript', 'c++', 'c#', 'go', 'rust', 'swift', 'kotlin']
        features['has_programming_language'] = 1 if any(lang in cv_skills_str for lang in prog_langs) else 0

        frameworks = ['django', 'fastapi', 'flask', 'spring', 'express']
        features['has_framework'] = 1 if any(fw in cv_skills_str for fw in frameworks) else 0

        # Databases (keep existing + specific booleans)
        databases = ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch']
        features['has_database'] = 1 if any(db in cv_skills_str for db in databases) else 0

        # Cloud & platform (separate flags)
        features['has_aws'] = 1 if 'aws' in cv_skills_str else 0
        features['has_azure'] = 1 if 'azure' in cv_skills_str else 0
        features['has_gcp'] = 1 if 'gcp' in cv_skills_str or 'google cloud' in cv_skills_str else 0
        features['has_docker'] = 1 if 'docker' in cv_skills_str else 0
        features['has_kubernetes'] = 1 if 'kubernetes' in cv_skills_str else 0

        # 3) Core BE Python JD flags (help ML see exact signals)
        features['has_python'] = 1 if 'python' in cv_skills_str else 0
        features['has_django'] = 1 if 'django' in cv_skills_str else 0
        features['has_fastapi'] = 1 if 'fastapi' in cv_skills_str else 0
        features['has_postgresql'] = 1 if 'postgresql' in cv_skills_str else 0
        features['has_redis'] = 1 if 'redis' in cv_skills_str else 0
        features['has_rest_api'] = 1 if 'rest api' in cv_skills_str else 0
        features['has_microservices'] = 1 if 'microservices' in cv_skills_str else 0
        features['has_git'] = 1 if 'git' in cv_skills_str else 0

        # 4) Skills text length
        features['skills_text_length'] = len(cv_skills_text or "")

        # 5) Trending skills (trim to backend-relevant)
        trending = ['kubernetes', 'elasticsearch', 'kafka', 'grpc', 'protobuf']
        num_trending = sum(1 for skill in trending if skill in cv_skills_str)
        features['num_trending_skills'] = num_trending

        # 6) Diversity & specificity
        features['skill_diversity_score'] = (len(set(cv_phrases)) / len(cv_phrases)) if cv_phrases else 0
        avg_skill_len = sum(len(s) for s in cv_phrases) / len(cv_phrases) if cv_phrases else 0
        features['avg_skill_length'] = round(avg_skill_len, 2)

        return features

    # --------------------------- Helpers ---------------------------
    def _normalize_phrase(self, s: str) -> str:
        if not s:
            return ""
        x = s.lower()
        x = x.replace("/", " ").replace("-", " ").replace("_", " ")
        x = re.sub(r"[.,;:]+", " ", x)
        x = re.sub(r"\s+", " ", x).strip()
        if x in self._skill_alias:
            x = self._skill_alias[x]
        return x

    def _normalize_token(self, w: str) -> str:
        if not w:
            return ""
        w = re.sub(r"[^a-z0-9+]", "", w.lower())
        if w in self._stem_map:
            return self._stem_map[w]
        if w in self._skill_alias:
            return self._skill_alias[w]
        return w

    def _expand_parent_skills(self, phrases: List[str]) -> List[str]:
        out = set(phrases)
        for parent, children in self._implies_parent.items():
            if any(child in out for child in children):
                out.add(parent)
        return list(out)

    def _expand_alternatives(self, s: str) -> List[str]:
        """
        Expand "OR" style entries: "Django or FastAPI" => ["django","fastapi"]; "MySQL/PostgreSQL" => ["mysql","postgresql"]
        """
        if not s:
            return []
        x = self._normalize_phrase(s)
        parts = re.split(r"\s+or\s+|/|\|", x)
        alts = [p.strip() for p in parts if p and p.strip()]
        # de-dup
        seen, out = set(), []
        for a in alts:
            if a not in seen:
                out.append(a)
                seen.add(a)
        return out if out else [x]

    def _extract_experience_features(self, cv_experience: str, jd_min_years: int,
                                     jd_responsibilities: List[str]) -> Dict:
        """
        Extract EXPERIENCE FEATURES (15 features)

        OUTPUT:
        {
            'total_years_experience': 6,        # CV has 6 years
            'years_ratio': 1.2,                 # 6/5 = 1.2
            'responsibilities_match': 0.67,     # 67% responsibilities matched
            'num_action_verbs': 8,             # "Led", "Built", etc.
            'has_metrics': 1,                   # Has numbers/percentages
            'num_metrics': 5,
            'has_leadership_keywords': 1,
            ...
        }
        """

        features = {}

        if not cv_experience:
            # Return zeros if no experience
            return self._get_empty_experience_features()

        cv_exp_lower = self._normalize_phrase(cv_experience)

        # Feature 1-2: Years of experience
        years = self._extract_years(cv_experience)
        features['total_years_experience'] = years
        features['years_ratio'] = round(years / jd_min_years, 2) if jd_min_years > 0 else 0

        # Feature 3-4: Responsibilities matching
        if jd_responsibilities:
            matched = 0
            for resp in jd_responsibilities:
                # normalize and extract keywords
                tokens = [self._normalize_token(w) for w in re.split(r"\s+", resp.lower())]
                keywords = [w for w in tokens if w and len(w) > 3]
                if not keywords:
                    continue
                found = sum(1 for kw in keywords if kw in cv_exp_lower)
                match_ratio = found / len(keywords)
                if match_ratio >= 0.4:
                    matched += 1

            resp_match_percentage = matched / len(jd_responsibilities)
            features['responsibilities_match_percentage'] = round(resp_match_percentage, 4)
            features['num_responsibilities_matched'] = matched
        else:
            features['responsibilities_match_percentage'] = 1.0
            features['num_responsibilities_matched'] = 0

        # Feature 5-7: Action verbs (achievement indicators)
        action_verbs = [
            'led', 'managed', 'built', 'developed', 'implemented', 'designed',
            'created', 'improved', 'optimized', 'achieved', 'delivered', 'launched'
        ]
        num_action_verbs = sum(cv_exp_lower.count(verb) for verb in action_verbs)
        features['num_action_verbs'] = num_action_verbs

        # Leadership verbs
        leadership_verbs = ['led', 'managed', 'coordinated', 'directed', 'supervised']
        num_leadership = sum(cv_exp_lower.count(verb) for verb in leadership_verbs)
        features['has_leadership_keywords'] = 1 if num_leadership > 0 else 0
        features['num_leadership_indicators'] = num_leadership

        # Feature 8-9: Quantifiable metrics
        metrics_pattern = r'\d+%|\d+\+|\d+[kKmM]|\d+ (users|customers|projects|teams)'
        metrics = re.findall(metrics_pattern, cv_experience)
        features['has_quantifiable_metrics'] = 1 if metrics else 0
        features['num_metrics'] = len(metrics)

        # Feature 10-11: Impact keywords
        impact_keywords = ['improved', 'increased', 'reduced', 'optimized', 'performance', 'efficiency']
        num_impact = sum(cv_exp_lower.count(kw) for kw in impact_keywords)
        features['num_impact_keywords'] = num_impact
        features['has_impact_language'] = 1 if num_impact > 0 else 0

        # Feature 12: Experience text length
        features['experience_text_length'] = len(cv_experience)

        # Feature 13-14: Company/scale indicators
        top_companies = ['google', 'microsoft', 'amazon', 'meta', 'facebook', 'apple', 'netflix']
        features['has_top_company'] = 1 if any(comp in cv_exp_lower for comp in top_companies) else 0

        scale_keywords = ['million', 'billion', 'large-scale', 'enterprise', '10m', '100m']
        features['has_large_scale_indicators'] = 1 if any(kw in cv_exp_lower for kw in scale_keywords) else 0

        # Feature 15: Promotion/growth indicators
        growth_keywords = ['promoted', 'advanced', 'progressed', 'senior']
        features['has_career_progression'] = 1 if any(kw in cv_exp_lower for kw in growth_keywords) else 0

        return features

    def _extract_years(self, experience_text: str) -> int:
        """Helper: Extract total years từ experience text"""
        # Pattern 1: "X year(s)"
        year_pattern = r'(\d+)\s*year\(?s?\)?'
        year_matches = re.findall(year_pattern, experience_text.lower())
        total_v1 = sum(int(y) for y in year_matches) if year_matches else 0

        # Pattern 2: "from YYYY to YYYY"
        date_pattern = r'from\s+(\d{4})\s+to\s+(\d{4})'
        date_matches = re.findall(date_pattern, experience_text)
        if date_matches:
            total_v2 = max(int(end) - int(start) for start, end in date_matches)
        else:
            total_v2 = 0

        return max(total_v1, total_v2)

    def _get_empty_experience_features(self) -> Dict:
        """Helper: Return zero features khi không có experience"""
        return {
            'total_years_experience': 0,
            'years_ratio': 0,
            'responsibilities_match_percentage': 0,
            'num_responsibilities_matched': 0,
            'num_action_verbs': 0,
            'has_leadership_keywords': 0,
            'num_leadership_indicators': 0,
            'has_quantifiable_metrics': 0,
            'num_metrics': 0,
            'num_impact_keywords': 0,
            'has_impact_language': 0,
            'experience_text_length': 0,
            'has_top_company': 0,
            'has_large_scale_indicators': 0,
            'has_career_progression': 0
        }

    def _extract_education_features(self, cv_education: str,
                                    jd_min_education: str) -> Dict:
        """
        Extract EDUCATION FEATURES (6 features)

        OUTPUT:
        {
            'degree_level': 2,                  # Bachelor = 2
            'degree_level_difference': 0,       # cv_level - jd_level
            'education_meets_requirement': 1,   # Binary
            'is_stem_degree': 1,               # CS/Engineering
            'education_text_length': 150,
            'has_top_university': 0
        }
        """

        features = {}

        # Degree levels
        degree_levels = {
            "phd": 4, "doctorate": 4,
            "master": 3, "mba": 3, "m.s.": 3, "m.a.": 3,
            "bachelor": 2, "b.s.": 2, "b.a.": 2, "b.e.": 2,
            "associate": 1, "diploma": 1
        }

        if cv_education:
            cv_edu_lower = cv_education.lower()

            # Feature 1: CV degree level
            cv_level = 0
            for degree, level in sorted(degree_levels.items(), key=lambda x: -x[1]):
                if degree in cv_edu_lower:
                    cv_level = level
                    break

            features['degree_level'] = cv_level

            # Feature 2-3: Comparison với JD
            jd_level = degree_levels.get(jd_min_education.lower(), 2)
            features['degree_level_difference'] = cv_level - jd_level
            features['education_meets_requirement'] = 1 if cv_level >= jd_level else 0

            # Feature 4: STEM degree
            stem_keywords = ['computer science', 'engineering', 'mathematics', 'science', 'technology']
            features['is_stem_degree'] = 1 if any(kw in cv_edu_lower for kw in stem_keywords) else 0

            # Feature 5: Text length
            features['education_text_length'] = len(cv_education)

            # Feature 6: Top university
            top_unis = ['stanford', 'mit', 'harvard', 'cambridge', 'oxford', 'berkeley', 'carnegie mellon', 'cmu']
            features['has_top_university'] = 1 if any(uni in cv_edu_lower for uni in top_unis) else 0
        else:
            # No education data
            features['degree_level'] = 0
            features['degree_level_difference'] = -2
            features['education_meets_requirement'] = 0
            features['is_stem_degree'] = 0
            features['education_text_length'] = 0
            features['has_top_university'] = 0

        return features

    def _extract_text_features(self, sections: Dict) -> Dict:
        """
        Extract TEXT QUALITY FEATURES (7 features)

        OUTPUT:
        {
            'cv_total_length': 1500,           # Total chars
            'about_text_length': 200,
            'num_sections_complete': 4,        # All 4 sections filled
            'has_complete_sections': 1,
            'about_quality_score': 0.75,      # Professional language
            'cv_completeness_score': 1.0,
            'avg_section_length': 375
        }
        """

        features = {}

        required_sections = ['ABOUT', 'SKILLS', 'WORK EXPERIENCE', 'EDUCATION']

        # Feature 1: Total CV length
        total_length = sum(len(sections.get(s, '')) for s in required_sections)
        features['cv_total_length'] = total_length

        # Feature 2: About section length
        about_text = sections.get('ABOUT', '')
        features['about_text_length'] = len(about_text)

        # Feature 3-4: Completeness
        complete_sections = sum(1 for s in required_sections if sections.get(s, '').strip())
        features['num_sections_complete'] = complete_sections
        features['has_complete_sections'] = 1 if complete_sections == 4 else 0

        # Feature 5: About quality (professional keywords)
        if about_text:
            prof_keywords = ['experience', 'skilled', 'proven', 'expert', 'proficient', 'specialized']
            about_lower = about_text.lower()
            prof_count = sum(1 for kw in prof_keywords if kw in about_lower)
            features['about_quality_score'] = min(prof_count / 3, 1.0)  # Normalize to 0-1
        else:
            features['about_quality_score'] = 0.0

        # Feature 6: Overall completeness score
        length_score = min(total_length / 1000, 1.0)  # Normalize: 1000 chars = 1.0
        section_score = complete_sections / 4
        features['cv_completeness_score'] = round((length_score + section_score) / 2, 4)

        # Feature 7: Average section length
        features['avg_section_length'] = int(total_length / 4) if total_length > 0 else 0

        return features


# =============================================================================
# EXAMPLE USAGE - CÁCH DÙNG MẪU
# =============================================================================

if __name__ == "__main__":
    """
    Ví dụ cách sử dụng FeatureExtractor

    CHẠY FILE NÀY ĐỂ TEST:
    python feature_extractor.py
    """

    # Example data
    cv = {
        "id": "cv_1.pdf",
        "sections": {
            "ABOUT": "Senior Software Engineer with 6 years experience",
            "SKILLS": "Python, Django, AWS, Docker, PostgreSQL",
            "WORK EXPERIENCE": "Built APIs for 3 years. Led team for 2 years. Worked at TechCorp from 2018 to 2024.",
            "EDUCATION": "B.S. in Computer Science from MIT, graduated 2018"
        }
    }

    jd = {
        "job_id": "JD_001",
        "title": "Backend Engineer",
        "requirements": {
            "skills": ["Python", "Django", "PostgreSQL", "AWS"],
            "min_years_experience": 5,
            "key_responsibilities": [
                "Build RESTful APIs",
                "Lead technical teams"
            ],
            "min_education": "Bachelor"
        }
    }

    # Extract features
    print("=" * 60)
    print("DEMO: FEATURE EXTRACTION")
    print("=" * 60)

    extractor = FeatureExtractor()
    features = extractor.extract_features(cv, jd)

    print(f"\nExtracted {len(features)} features:")
    print(f"\nFirst 10 features:")
    for i, (name, value) in enumerate(zip(extractor.feature_names[:10], features[:10]), 1):
        print(f"  {i}. {name}: {value}")

    print(f"\n... (total {len(features)} features)")

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETED!")
    print("=" * 60)
    print(f"\nThese {len(features)} numbers will be used to train ML model")
