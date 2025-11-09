"""
FILE: inference.py
PURPOSE: Score new CVs using trained model
=============================================================================
HƯỚNG DẪN SỬ DỤNG:
=============================================================================

FILE NÀY LÀM GÌ?
- Dùng trained model để chấm điểm CV mới
- Combine rule-based + ML predictions
- Cho ra final score + rating

CÁCH DÙNG (CÓ 3 CÁCH):

CÁCH 1 - Score 1 CV với 1 JD:
    python inference.py --cv data/Segmented_Text_2/cv_001.json --jd data/sample_jds/jd_backend.json

CÁCH 2 - Score nhiều CVs với 1 JD:
    python inference.py --cv_folder data/Segmented_Text_2 --jd data/sample_jds/jd_backend.json

CÁCH 3 - Dùng trong Python code:
    from inference import CVScorer
    scorer = CVScorer()
    result = scorer.score(cv_data, jd_data)

OUTPUT:
- Console: In ra scores
- File: results/scoring_results.json

LƯU Ý:
- Phải train model trước (chạy train_model.py)
- Model files phải có trong folder models/

=============================================================================
"""

import os
import json
import joblib
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List

# Import từ files khác
from cv_jd_matcher import SimpleCVJDMatcher
from feature_extractor import FeatureExtractor


class CVScorer:
    """
    Class để score CVs using trained model

    Workflow:
    1. Load trained model (one-time)
    2. For each CV-JD pair:
       - Calculate rule-based score
       - Extract features
       - Predict ML score
       - Combine scores
    """

    def __init__(self, model_folder='models', rule_weight: float = 0.45):
        """
        Khởi tạo scorer

        CÁCH DÙNG:
            scorer = CVScorer()  # Dùng default folder

            hoặc:

            scorer = CVScorer(model_folder='path/to/models')

        PARAMS:
            model_folder: Folder chứa trained model files
            rule_weight: Weight for rule-based score in ensemble (0.0-1.0)
        """
        self.model_folder = model_folder
        self.rule_weight = rule_weight

        # Load trained artifacts
        print("Loading trained model...")
        self._load_model()

        # Initialize components
        self.matcher = SimpleCVJDMatcher()
        self.extractor = FeatureExtractor()

        print("✓ Scorer ready!")

    def _load_model(self):
        """
        Load trained model và artifacts

        FILES NEEDED:
        - trained_model.pkl
        - scaler.pkl
        - feature_names.json
        """
        try:
            # Load model
            model_path = os.path.join(self.model_folder, 'trained_model.pkl')
            self.model = joblib.load(model_path)
            print(f"  ✓ Model loaded from {model_path}")

            # Load scaler
            scaler_path = os.path.join(self.model_folder, 'scaler.pkl')
            self.scaler = joblib.load(scaler_path)
            print(f"  ✓ Scaler loaded from {scaler_path}")

            # Load feature names
            feature_names_path = os.path.join(self.model_folder, 'feature_names.json')
            with open(feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
            print(f"  ✓ Feature names loaded ({len(self.feature_names)} features)")

        except FileNotFoundError as e:
            print(f"\n✗ ERROR: Could not load model files!")
            print(f"  {e}")
            print(f"\n  Please run train_model.py first to train the model.")
            print(f"  Expected files in '{self.model_folder}':")
            print(f"    - trained_model.pkl")
            print(f"    - scaler.pkl")
            print(f"    - feature_names.json")
            raise

    def score(self, cv_data: Dict, jd_data: Dict) -> Dict:
        """
        HÀM CHÍNH - Score a single CV against a JD

        CÁCH DÙNG:
            result = scorer.score(cv_data, jd_data)
            print(f"Score: {result['final_score']}")

        INPUT:
            cv_data: CV dictionary (from Task 2 NER)
            jd_data: JD dictionary

        OUTPUT:
            Dictionary với scores và details:
            {
                'cv_id': 'cv_001.pdf',
                'jd_id': 'JD_001',
                'rule_based_score': 68.5,
                'ml_predicted_score': 72.3,
                'final_score': 70.8,
                'confidence': 85.2,
                'rating': 'Very Good',
                'recommendation': 'Proceed to interview',
                'breakdown': {...}
            }
        """

        # STAGE 1: Rule-based scoring
        rule_result = self.matcher.score_cv_against_jd(cv_data, jd_data)
        rule_score = rule_result['total_score']

        # STAGE 2: ML prediction
        features = self.extractor.extract_features(cv_data, jd_data)
        features_scaled = self.scaler.transform([features])
        ml_score = self.model.predict(features_scaled)[0]

        # Calculate confidence
        # Dùng variance của tree predictions
        confidence = self._calculate_confidence(features_scaled)

        # ENSEMBLE: Combine rule-based + ML
        # Use configurable weights
        rw = getattr(self, "rule_weight", 0.45)
        mw = 1.0 - rw
        final_score = rw * rule_score + mw * ml_score

        # Rating và recommendation
        rating = self._get_rating(final_score)
        recommendation = self._get_recommendation(final_score, confidence)

        return {
            'cv_id': cv_data.get('id', 'Unknown'),
            'jd_id': jd_data.get('job_id', 'Unknown'),
            'jd_title': jd_data.get('title', 'Unknown'),
            'rule_based_score': round(rule_score, 2),
            'ml_predicted_score': round(ml_score, 2),
            'final_score': round(final_score, 2),
            'confidence': round(confidence, 2),
            'rating': rating,
            'recommendation': recommendation,
            'breakdown': rule_result['breakdown'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def _calculate_confidence(self, features_scaled: np.ndarray) -> float:
        """
        Calculate prediction confidence

        METHOD:
        - Get predictions từ all 100 trees
        - Low variance = high confidence
        - High variance = low confidence

        RETURN:
            Confidence score (0-100)
        """
        # Get predictions từ all trees
        tree_predictions = np.array([
            tree.predict(features_scaled)[0]
            for tree in self.model.estimators_
        ])

        # Calculate variance
        std = np.std(tree_predictions)
        mean = np.mean(tree_predictions)

        # Convert to confidence (lower std = higher confidence)
        # Normalize: std of 0 = 100% confidence, std of 10+ = 0% confidence
        confidence = max(0, 100 - (std / mean * 100)) if mean > 0 else 50

        return confidence

    def _get_rating(self, score: float) -> str:
        """Convert score to rating"""
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

    def _get_recommendation(self, score: float, confidence: float) -> str:
        """Get hiring recommendation"""
        if score >= 85 and confidence >= 80:
            return '🚀 Fast-track to interview - Excellent candidate'
        elif score >= 70 and confidence >= 75:
            return '⭐ Priority interview - Strong candidate'
        elif score >= 60 and confidence >= 70:
            return '✅ Standard interview - Qualified candidate'
        elif score >= 50:
            return '⚠️ Review carefully - Some gaps present'
        elif score >= 40:
            return '🔍 Consider only if desperate - Many gaps'
        else:
            return '❌ Not a good fit - Reject'

    def score_multiple(self, cv_folder: str, jd_data: Dict,
                       output_file: str = 'results/scoring_results.json') -> List[Dict]:
        """
        Score nhiều CVs với 1 JD

        CÁCH DÙNG:
            results = scorer.score_multiple(
                cv_folder='data/Segmented_Text_2',
                jd_data=my_jd
            )

        INPUT:
            cv_folder: Folder chứa CV files
            jd_data: JD dictionary
            output_file: File để save results

        RETURN:
            List of scoring results
        """
        print(f"\n{'=' * 70}")
        print(f"SCORING MULTIPLE CVS")
        print(f"{'=' * 70}")
        print(f"CV Folder: {cv_folder}")
        print(f"JD: {jd_data.get('title', 'Unknown')}")
        print(f"{'=' * 70}\n")

        # Load all CVs
        cv_files = [f for f in os.listdir(cv_folder) if f.endswith('.json')]
        total_cvs = len(cv_files)

        print(f"Found {total_cvs} CVs to score...")

        results = []

        for i, cv_file in enumerate(cv_files, 1):
            try:
                # Load CV
                with open(os.path.join(cv_folder, cv_file), 'r', encoding='utf-8') as f:
                    cv_data = json.load(f)

                # Score
                result = self.score(cv_data, jd_data)
                results.append(result)

                # Progress
                if i % 10 == 0 or i == total_cvs:
                    print(f"  Progress: {i}/{total_cvs} CVs scored...")

            except Exception as e:
                print(f"  Warning: Could not score {cv_file}: {e}")
                continue

        # Sort by final score (descending)
        results.sort(key=lambda x: x['final_score'], reverse=True)

        # Save results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Scored {len(results)} CVs")
        print(f"✓ Results saved to: {output_file}")

        # Print summary
        self._print_summary(results, jd_data)

        return results

    def _print_summary(self, results: List[Dict], jd_data: Dict):
        """Print scoring summary"""
        print(f"\n{'=' * 70}")
        print(f"SCORING SUMMARY")
        print(f"{'=' * 70}")
        print(f"Job: {jd_data.get('title', 'Unknown')}")
        rw = getattr(self, "rule_weight", 0.45)
        print(f"Ensemble Weights: rule={rw:.2f}, ml={1-rw:.2f}")
        print(f"Total CVs Scored: {len(results)}")

        if results:
            scores = [r['final_score'] for r in results]
            print(f"\nScore Statistics:")
            print(f"  Average:  {np.mean(scores):.2f}")
            print(f"  Median:   {np.median(scores):.2f}")
            print(f"  Min:      {np.min(scores):.2f}")
            print(f"  Max:      {np.max(scores):.2f}")

            # Rating distribution
            ratings = {}
            for r in results:
                rating = r['rating']
                ratings[rating] = ratings.get(rating, 0) + 1

            print(f"\nRating Distribution:")
            for rating in ['Excellent', 'Very Good', 'Good', 'Fair', 'Below Average', 'Poor']:
                count = ratings.get(rating, 0)
                percentage = count / len(results) * 100
                print(f"  {rating:15s}: {count:3d} ({percentage:5.1f}%)")

            # Top candidates
            print(f"\nTop 5 Candidates:")
            for i, r in enumerate(results[:5], 1):
                print(f"  {i}. {r['cv_id']:20s} Score: {r['final_score']:5.1f} ({r['rating']})")

        print(f"{'=' * 70}\n")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """
    Main function cho command line usage

    EXAMPLES:

    Score 1 CV:
        python inference.py --cv cv_001.json --jd jd_backend.json

    Score folder of CVs:
        python inference.py --cv_folder data/Segmented_Text_2 --jd jd_backend.json
    """

    parser = argparse.ArgumentParser(
        description='Score CVs against Job Descriptions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Score a single CV
  python inference.py --cv data/Segmented_Text_2/cv_001.json --jd data/sample_jds/jd_backend.json

  # Score multiple CVs
  python inference.py --cv_folder data/Segmented_Text_2 --jd data/sample_jds/jd_backend.json

  # Specify output file
  python inference.py --cv_folder data/Segmented_Text_2 --jd data/sample_jds/jd_backend.json --output results/my_results.json
        """
    )

    parser.add_argument('--cv', type=str, help='Path to single CV file')
    parser.add_argument('--cv_folder', type=str, help='Path to folder with multiple CVs')
    parser.add_argument('--jd', type=str, required=True, help='Path to JD file')
    parser.add_argument('--output', type=str, default='results/scoring_results.json',
                        help='Output file for results (default: results/scoring_results.json)')
    parser.add_argument('--model_folder', type=str, default='models',
                        help='Folder containing trained model (default: models)')
    parser.add_argument('--rule_weight', type=float, default=0.45,
                        help='Weight for rule-based score in final ensemble (0.0–1.0). Default: 0.45')

    args = parser.parse_args()

    # Validate arguments
    if not args.cv and not args.cv_folder:
        parser.error("Must specify either --cv or --cv_folder")

    if args.cv and args.cv_folder:
        parser.error("Cannot specify both --cv and --cv_folder")
    if not (0.0 <= args.rule_weight <= 1.0):
        parser.error("--rule_weight must be between 0.0 and 1.0")

    # Load JD
    print(f"Loading JD from {args.jd}...")
    try:
        with open(args.jd, 'r', encoding='utf-8') as f:
            jd_data = json.load(f)
        print(f"✓ JD loaded: {jd_data.get('title', 'Unknown')}")
    except Exception as e:
        print(f"✗ Error loading JD: {e}")
        return

    # Initialize scorer
    scorer = CVScorer(model_folder=args.model_folder, rule_weight=args.rule_weight)

    # Score based on input type
    if args.cv:
        # Single CV
        print(f"\nLoading CV from {args.cv}...")
        try:
            with open(args.cv, 'r', encoding='utf-8') as f:
                cv_data = json.load(f)
            print(f"✓ CV loaded: {cv_data.get('id', 'Unknown')}")
        except Exception as e:
            print(f"✗ Error loading CV: {e}")
            return

        # Score
        result = scorer.score(cv_data, jd_data)

        # Print result
        print(f"\n{'=' * 70}")
        print(f"SCORING RESULT")
        print(f"{'=' * 70}")
        print(f"CV:  {result['cv_id']}")
        print(f"JD:  {result['jd_title']}")
        print(f"\n{'Score Breakdown:':^70}")
        print(f"{'-' * 70}")
        print(f"  Rule-Based Score:     {result['rule_based_score']:5.1f}/100")
        print(f"  ML Predicted Score:   {result['ml_predicted_score']:5.1f}/100")
        print(f"  Final Score:          {result['final_score']:5.1f}/100")
        print(f"  Confidence:           {result['confidence']:5.1f}%")
        print(f"  Ensemble Weights:     rule={args.rule_weight:.2f}  ml={1-args.rule_weight:.2f}")
        print(f"{'-' * 70}")
        print(f"  Rating:               {result['rating']}")
        print(f"  Recommendation:       {result['recommendation']}")
        print(f"{'=' * 70}\n")

        # Save to file
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"✓ Result saved to: {args.output}\n")

    else:
        # Multiple CVs
        results = scorer.score_multiple(args.cv_folder, jd_data, args.output)


# =============================================================================
# EXAMPLE USAGE IN CODE
# =============================================================================

def example_usage():
    """
    Ví dụ cách dùng CVScorer trong Python code

    KHÔNG PHẢI CLI - Dùng khi import vào code khác
    """

    # Example CV và JD
    cv = {
        "id": "cv_example.pdf",
        "sections": {
            "ABOUT": "Senior Backend Engineer with 6 years experience",
            "SKILLS": "Python, Django, PostgreSQL, AWS, Docker",
            "WORK EXPERIENCE": "Built APIs for 3 years. Led team for 2 years.",
            "EDUCATION": "B.S. in Computer Science"
        }
    }

    jd = {
        "job_id": "JD_001",
        "title": "Senior Backend Engineer",
        "requirements": {
            "skills": ["Python", "Django", "PostgreSQL", "AWS"],
            "min_years_experience": 5,
            "key_responsibilities": ["Build APIs", "Lead teams"],
            "min_education": "Bachelor"
        }
    }

    # Initialize scorer
    scorer = CVScorer()

    # Score
    result = scorer.score(cv, jd)

    # Use result
    print(f"Score: {result['final_score']}")
    print(f"Rating: {result['rating']}")

    if result['final_score'] >= 70:
        print("✓ Recommend for interview!")
    else:
        print("✗ Not a good fit")


if __name__ == "__main__":
    """
    Entry point khi chạy file directly

    COMMAND:
    python inference.py --cv cv_file.json --jd jd_file.json
    """
    try:
        # Nếu không có tham số, chạy example
        import sys
        if len(sys.argv) == 1:
            example_usage()
        else:
            main()
    except Exception as e:
        print(f"Error: {e}")
