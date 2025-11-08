"""
FILE: train_model.py
PURPOSE: Train ML model để chấm điểm CV

=============================================================================
HƯỚNG DẪN SỬ DỤNG:
=============================================================================

FILE NÀY LÀM GÌ?
- Train (huấn luyện) ML model từ 500 CVs
- Dùng rule-based scores làm labels
- Save trained model để dùng sau

CÁCH CHẠY (RẤT ĐỠN GIẢN):
1. Mở Terminal/Command Prompt
2. Gõ: python train_model.py
3. Đợi 2-5 phút
4. Xong! Model đã được lưu

INPUT CẦN CÓ:
- 500 CVs trong folder: data/Segmented_Text_2/
- 10 JDs trong folder: data/sample_jds/

OUTPUT:
- models/trained_model.pkl (trained model)
- models/scaler.pkl (feature scaler)
- models/feature_names.json (feature list)
- results/training_report.json (metrics)

THỜI GIAN:
- Với 500 CVs × 10 JDs = 5,000 samples
- Training time: 2-5 phút tùy máy

LƯU Ý:
- Chỉ cần chạy 1 LẦN để train
- Sau đó dùng trained model mãi mãi
- Muốn retrain? Chạy lại file này

=============================================================================
"""

import os
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import datetime

# Import từ files khác
from cv_jd_matcher import SimpleCVJDMatcher
from feature_extractor import FeatureExtractor


class ModelTrainer:
    """
    Class để train ML model

    Workflow:
    1. Load CVs và JDs
    2. Generate training data (rule-based scores + features)
    3. Train Random Forest model
    4. Evaluate model
    5. Save model
    """

    def __init__(self, cvs_folder, jds_folder, output_folder):
        """
        Khởi tạo trainer

        CÁCH DÙNG:
            trainer = ModelTrainer(
                cvs_folder="data/Segmented_Text_2",
                jds_folder="data/sample_jds",
                output_folder="models"
            )

        PARAMS:
            cvs_folder: Folder chứa 500 CV files
            jds_folder: Folder chứa 10 JD files
            output_folder: Folder để save model
        """
        self.cvs_folder = cvs_folder
        self.jds_folder = jds_folder
        self.output_folder = output_folder

        # Tạo output folder nếu chưa có
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs("results", exist_ok=True)

        # Initialize components
        self.matcher = SimpleCVJDMatcher()
        self.extractor = FeatureExtractor()

        print("=" * 70)
        print("MODEL TRAINER INITIALIZED")
        print("=" * 70)
        print(f"CVs folder: {cvs_folder}")
        print(f"JDs folder: {jds_folder}")
        print(f"Output folder: {output_folder}")
        print("=" * 70)

    def load_data(self):
        """
        Load CVs và JDs từ folders

        RETURN:
            cvs: List of CV dictionaries
            jds: List of JD dictionaries
        """
        print("\n[STEP 1/6] Loading data...")

        # Load CVs
        cvs = []
        cv_files = [f for f in os.listdir(self.cvs_folder) if f.endswith('.json')]

        for cv_file in cv_files:
            try:
                with open(os.path.join(self.cvs_folder, cv_file), 'r', encoding='utf-8') as f:
                    cv_data = json.load(f)
                    cvs.append(cv_data)
            except Exception as e:
                print(f"  Warning: Could not load {cv_file}: {e}")

        print(f"  ✓ Loaded {len(cvs)} CVs")

        # Load JDs
        jds = []
        jd_files = [f for f in os.listdir(self.jds_folder) if f.endswith('.json')]

        for jd_file in jd_files:
            try:
                with open(os.path.join(self.jds_folder, jd_file), 'r', encoding='utf-8') as f:
                    jd_data = json.load(f)
                    jds.append(jd_data)
            except Exception as e:
                print(f"  Warning: Could not load {jd_file}: {e}")

        print(f"  ✓ Loaded {len(jds)} JDs")

        if len(cvs) == 0 or len(jds) == 0:
            raise ValueError("No CVs or JDs loaded! Check your data folders.")

        return cvs, jds

    def generate_training_data(self, cvs, jds):
        """
        Generate training data từ CVs + JDs

        PROCESS:
        - Với mỗi CV-JD pair:
          1. Calculate rule-based score (label)
          2. Extract features (input)

        RETURN:
            X: Features array (N × 40)
            y: Labels array (N × 1)
            N = số CVs × số JDs
        """
        print("\n[STEP 2/6] Generating training data...")
        print(f"  Creating {len(cvs)} CVs × {len(jds)} JDs = {len(cvs) * len(jds)} samples")

        X_list = []
        y_list = []

        total_pairs = len(cvs) * len(jds)
        processed = 0

        for cv in cvs:
            for jd in jds:
                try:
                    # Rule-based score (label)
                    rule_result = self.matcher.score_cv_against_jd(cv, jd)
                    y_list.append(rule_result['total_score'])

                    # Features (input)
                    features = self.extractor.extract_features(cv, jd)
                    X_list.append(features)

                    processed += 1

                    # Progress indicator
                    if processed % 500 == 0:
                        print(f"    Processed {processed}/{total_pairs} pairs...")

                except Exception as e:
                    print(f"  Warning: Error processing CV {cv.get('id')}: {e}")
                    continue

        # Convert to numpy arrays
        X = np.array(X_list)
        y = np.array(y_list)

        print(f"  ✓ Generated {len(X)} training samples")
        print(f"    X shape: {X.shape} (samples × features)")
        print(f"    y shape: {y.shape} (samples)")

        return X, y

    def split_and_scale(self, X, y):
        """
        Split data và scale features

        SPLIT:
        - 80% training
        - 20% testing

        SCALING:
        - Normalize features về 0-1 range
        - Giúp model learn tốt hơn

        RETURN:
            X_train_scaled, X_test_scaled, y_train, y_test, scaler
        """
        print("\n[STEP 3/6] Splitting and scaling data...")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,  # 20% for testing
            random_state=42  # Reproducibility
        )

        print(f"  ✓ Train set: {len(X_train)} samples")
        print(f"  ✓ Test set: {len(X_test)} samples")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"  ✓ Features scaled (mean=0, std=1)")

        return X_train_scaled, X_test_scaled, y_train, y_test, scaler

    def train_model(self, X_train, y_train):
        """
        Train Random Forest model

        MODEL:
        - Algorithm: Random Forest Regressor
        - Trees: 100
        - Max depth: 10 (prevent overfitting)

        RETURN:
            trained model
        """
        print("\n[STEP 4/6] Training Random Forest model...")
        print("  Parameters:")
        print("    - n_estimators: 100 trees")
        print("    - max_depth: 10")
        print("    - min_samples_split: 5")

        model = RandomForestRegressor(
            n_estimators=100,  # 100 decision trees
            max_depth=10,  # Max tree depth (prevent overfitting)
            min_samples_split=5,  # Min samples to split node
            random_state=42,  # Reproducibility
            n_jobs=-1,  # Use all CPU cores
            verbose=0
        )

        print("  Training... (this may take 2-5 minutes)")
        model.fit(X_train, y_train)

        print("  ✓ Model trained successfully!")

        return model

    def evaluate_model(self, model, X_train, X_test, y_train, y_test):
        """
        Evaluate model performance

        METRICS:
        - R² Score: How well model explains variance (higher = better)
        - MAE: Mean Absolute Error (lower = better)
        - RMSE: Root Mean Squared Error (lower = better)
        - Cross-validation R²: Generalization performance

        RETURN:
            metrics dictionary
        """
        print("\n[STEP 5/6] Evaluating model...")

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            'train_r2': r2_score(y_train, y_train_pred),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_r2': r2_score(y_test, y_test_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred))
        }

        # Cross-validation
        print("  Running 5-fold cross-validation...")
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=5, scoring='r2', n_jobs=-1)
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()

        # Print results
        print("\n  " + "=" * 60)
        print("  TRAINING RESULTS")
        print("  " + "=" * 60)
        print(f"  Train R² Score:  {metrics['train_r2']:.4f} (explains {metrics['train_r2'] * 100:.1f}% variance)")
        print(f"  Train MAE:       {metrics['train_mae']:.2f} points")
        print(f"  Train RMSE:      {metrics['train_rmse']:.2f} points")
        print()
        print(f"  Test R² Score:   {metrics['test_r2']:.4f} (explains {metrics['test_r2'] * 100:.1f}% variance)")
        print(f"  Test MAE:        {metrics['test_mae']:.2f} points")
        print(f"  Test RMSE:       {metrics['test_rmse']:.2f} points")
        print()
        print(f"  Cross-Val R²:    {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
        print("  " + "=" * 60)

        # Interpretation
        print("\n  INTERPRETATION:")
        if metrics['test_r2'] >= 0.85:
            print("  ✓ EXCELLENT: Model explains 85%+ of variance")
        elif metrics['test_r2'] >= 0.75:
            print("  ✓ GOOD: Model explains 75-85% of variance")
        elif metrics['test_r2'] >= 0.65:
            print("  ⚠ FAIR: Model explains 65-75% of variance")
        else:
            print("  ✗ POOR: Model explains <65% of variance - need improvement")

        if metrics['test_mae'] <= 5:
            print("  ✓ EXCELLENT: Average error ≤5 points")
        elif metrics['test_mae'] <= 8:
            print("  ✓ GOOD: Average error 5-8 points")
        else:
            print("  ⚠ FAIR: Average error >8 points")

        return metrics

    def save_model(self, model, scaler, metrics):
        """
        Save trained model và artifacts

        FILES:
        - trained_model.pkl: Trained Random Forest model
        - scaler.pkl: Feature scaler
        - feature_names.json: List of feature names
        - training_report.json: Metrics và info
        """
        print("\n[STEP 6/6] Saving model and artifacts...")

        # Save model
        model_path = os.path.join(self.output_folder, 'trained_model.pkl')
        joblib.dump(model, model_path)
        print(f"  ✓ Model saved: {model_path}")

        # Save scaler
        scaler_path = os.path.join(self.output_folder, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"  ✓ Scaler saved: {scaler_path}")

        # Save feature names
        feature_names_path = os.path.join(self.output_folder, 'feature_names.json')
        with open(feature_names_path, 'w') as f:
            json.dump(self.extractor.feature_names, f, indent=2)
        print(f"  ✓ Feature names saved: {feature_names_path}")

        # Save training report
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': metrics,
            'model_params': {
                'algorithm': 'RandomForestRegressor',
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5
            },
            'feature_count': len(self.extractor.feature_names),
            'feature_names': self.extractor.feature_names
        }

        report_path = os.path.join('results', 'training_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"  ✓ Training report saved: {report_path}")

        # Get feature importance
        feature_importance = sorted(
            zip(self.extractor.feature_names, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )

        importance_path = os.path.join('results', 'feature_importance.json')
        with open(importance_path, 'w') as f:
            json.dump([
                {'feature': name, 'importance': float(imp)}
                for name, imp in feature_importance
            ], f, indent=2)
        print(f"  ✓ Feature importance saved: {importance_path}")

        # Print top 10 important features
        print("\n  TOP 10 MOST IMPORTANT FEATURES:")
        for i, (name, imp) in enumerate(feature_importance[:10], 1):
            print(f"    {i:2d}. {name:40s} {imp:.4f}")

    def run_full_training(self):
        """
        HÀM CHÍNH - Chạy toàn bộ training pipeline

        CÁCH DÙNG:
            trainer = ModelTrainer(...)
            trainer.run_full_training()

        Hàm này sẽ tự động:
        1. Load data
        2. Generate training samples
        3. Split and scale
        4. Train model
        5. Evaluate
        6. Save everything
        """
        start_time = datetime.now()

        print("\n")
        print("╔" + "=" * 68 + "╗")
        print("║" + " " * 20 + "TRAINING START" + " " * 34 + "║")
        print("╚" + "=" * 68 + "╝")

        try:
            # Step 1: Load data
            cvs, jds = self.load_data()

            # Step 2: Generate training data
            X, y = self.generate_training_data(cvs, jds)

            # Step 3: Split and scale
            X_train, X_test, y_train, y_test, scaler = self.split_and_scale(X, y)

            # Step 4: Train model
            model = self.train_model(X_train, y_train)

            # Step 5: Evaluate
            metrics = self.evaluate_model(model, X_train, X_test, y_train, y_test)

            # Step 6: Save
            self.save_model(model, scaler, metrics)

            # Success!
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            print("\n")
            print("╔" + "=" * 68 + "╗")
            print("║" + " " * 20 + "TRAINING COMPLETED!" + " " * 29 + "║")
            print("╚" + "=" * 68 + "╝")
            print(f"\n  Total time: {duration:.1f} seconds ({duration / 60:.1f} minutes)")
            print(f"\n  Model ready to use!")
            print(f"  Next step: Use inference.py to score new CVs")
            print()

            return model, scaler, metrics

        except Exception as e:
            print("\n")
            print("╔" + "=" * 68 + "╗")
            print("║" + " " * 20 + "TRAINING FAILED!" + " " * 31 + "║")
            print("╚" + "=" * 68 + "╝")
            print(f"\n  Error: {e}")
            print(f"\n  Please check:")
            print(f"    1. CVs folder exists: {self.cvs_folder}")
            print(f"    2. JDs folder exists: {self.jds_folder}")
            print(f"    3. Files are valid JSON format")
            print()
            raise


# =============================================================================
# MAIN EXECUTION - CHẠY TRAINING
# =============================================================================

def main():
    """
    Hàm main - Điểm bắt đầu của chương trình

    CÁCH CHẠY:
    python train_model.py

    Hoặc tùy chỉnh paths:
    python train_model.py --cvs_folder path/to/cvs --jds_folder path/to/jds
    """

    # Paths (có thể thay đổi nếu cần)
    CVS_FOLDER = "data/Segmented_Text_2"
    JDS_FOLDER = "data/test_jd_new"
    OUTPUT_FOLDER = "models"

    # Create trainer
    trainer = ModelTrainer(
        cvs_folder=CVS_FOLDER,
        jds_folder=JDS_FOLDER,
        output_folder=OUTPUT_FOLDER
    )

    # Run training
    trainer.run_full_training()


if __name__ == "__main__":
    """
    Chạy khi file được execute directly

    COMMAND:
    python train_model.py
    """
    main()
