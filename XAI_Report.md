# Explainable AI (XAI) Report
### Project: AI-based CV Evaluation System  
**Team Members:** Thịnh, Phát, Thông, Hương, An, Tiến  
**Date:** November 12, 2025  

---

## 1. Overview

This section provides an explainable evaluation of the trained Random Forest model used to score candidate CVs.  
The model was developed using structured and cleaned text features extracted from job descriptions (JDs) and candidate resumes (CVs).  
Explainable AI (XAI) techniques are applied to interpret **how and why** the model generates specific predictions, ensuring **transparency, fairness, and accountability**.

---

## 2. Model Evaluation Summary

|      Metric      |   Value   |                       Interpretation                                |
|------------------|-----------|---------------------------------------------------------------------|
|   **R² (Test)**  |   0.824   | Model explains **~82% of the variance** in unseen data (Excellent). |
|   **MAE**        |   2.07    | Average absolute error ≈ **2 points**, indicating high accuracy.    |
|   **RMSE**       |   3.28    | Low overall prediction deviation.                                   |

- The model generalizes well across different CVs.  
- Residuals are symmetrically centered around 0, indicating no systemic bias.  
- Top-ranked features align with realistic HR evaluation criteria (skills, experience, education).

---

## 3. Global Explainability (Model-Level)

At the **model level**, we analyzed **feature importance** using the Random Forest’s internal weights.  
The results show that the model relies on interpretable and relevant features for predicting CV scores.

**Top 10 influential features:**
1. `skills_match_percentage`  
2. `total_years_experience`  
3. `responsibilities_match_percentage`  
4. `education_meets_requirement`  
5. `num_skills_matched`  
6. `degree_level_difference`  
7. `years_ratio`  
8. `education_text_length`  
9. `num_responsibilities_matched`  
10. `is_stem_degree`

 **Interpretation:**  
The model assigns higher weight to the matching rate between skills and job requirements, confirming that **technical competency** and **experience alignment** play a major role in candidate evaluation.

---

## 4. Global Explainability (Data-Level via SHAP)

We employed **SHAP (SHapley Additive exPlanations)** to provide a more rigorous, data-level interpretation of model behavior.  
SHAP quantifies the marginal contribution of each feature to the predicted score for every individual CV.

**Key observations from the SHAP Summary Plot:**
- Red dots (high feature values) push predictions **upward** (increase CV score).  
- Blue dots (low feature values) push predictions **downward** (decrease CV score).  
- Consistent feature importance ranking between SHAP and Random Forest confirms **model stability**.

 **Conclusion:**  
The SHAP global analysis demonstrates that the model’s decision patterns are interpretable and aligned with human recruitment logic — high experience, high skill matching, and relevant education yield higher predicted scores.

---

## 5. Local Explainability (Individual CV Analysis)

To examine the explainability for a **single candidate**, we used **SHAP Waterfall Plot** (e.g., sample index 0).  
Each bar represents the contribution of one feature to the final predicted score relative to the model’s baseline expectation.

**Example interpretation:**
- `degree_level_difference (+1.55)` and `education_meets_requirement (+1.06)` significantly **increase** the score.  
- `skills_match_percentage (−1.50)` **decreases** the score due to partial mismatch between skills and job requirements.  
- Other features such as `years_ratio` and `num_skills_matched` contribute moderately.

 **Summary:**  
The candidate’s final predicted score of **~37** results from a balanced trade-off between high education relevance and limited skill matching.  
This level of transparency allows HR reviewers to **understand and trust** AI-driven evaluations.

---

## 6. Ethical and Practical Implications

The XAI component plays a crucial role in:
- **Transparency:** Every prediction can be justified via interpretable factors.  
- **Fairness:** Prevents hidden bias by showing explicit reasons for high/low scores.  
- **Accountability:** Helps explain decisions to HR managers and applicants.  
- **Improvement:** Feedback from SHAP insights can refine feature engineering and data labeling.

---

## 7. Conclusion

The integration of Explainable AI techniques — both **feature importance** and **SHAP analysis** — provides a comprehensive understanding of model behavior in the CV scoring system.

>  The Random Forest model achieves strong predictive performance (R² = 0.824, MAE = 2.07).  
>  Global explainability reveals HR-aligned logic behind predictions.  
>  Local explainability ensures decision traceability for each CV.  

Hence, the system is not only **accurate** but also **transparent**, ensuring reliable support for AI-assisted recruitment decisions within the organization.

---

**End of XAI Report**

