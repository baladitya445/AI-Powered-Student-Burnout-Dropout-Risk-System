# 🎓 AI-Powered Student Burnout & Dropout Risk System
## 📌 Project Overview
Student burnout and dropout risk are critical challenges in higher education. Traditional detection mechanisms rely on reactive indicators like failing grades or chronic absenteeism. However, behavioural disengagement signals often emerge much earlier. 

This project is a **proactive behavioural risk intelligence platform** built to detect early signals of academic burnout. Using a Random Forest classifier deployed via a Streamlit web application, the system analyzes student engagement metrics and categorizes them into Low, Medium, or High risk levels to enable timely, structured interventions.

## ✨ Key Features
* **Early Warning Detection:** Analyzes behavioral signals (LMS logins, assignment delays, sentiment, attendance) before academic failure occurs.
* **Continuous Risk Scoring:** Quantifies burnout risk on a granular 0–100 scale.
* **Interactive Dashboard:** Built with Streamlit, providing an intuitive interface for educators to monitor cohort health.
* **Explainable AI:** Highlights key behavioural drivers (e.g., assignment delay, negative sentiment) using feature importance metrics.

## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn (Random Forest Classifier)
* **Data Processing:** Pandas, NumPy
* **Frontend/Deployment:** Streamlit

# 📊 Dataset Information

## Dataset Type
**Synthetic Dataset**

## Why Synthetic Data?
Due to the absence of publicly available, ethically shareable student behavioural datasets (containing LMS activity, sentiment indicators, and attendance records), a synthetic dataset was generated to simulate realistic academic engagement patterns.

Real student behavioural data is:
* Institutionally protected.
* Privacy-sensitive (FERPA/GDPR constraints).
* Not publicly accessible for experimentation.

Therefore, synthetic data was engineered to replicate plausible behavioural burnout dynamics while preserving structural realism.

## How the Dataset Was Generated
The dataset was generated programmatically using controlled statistical distributions and behavioural assumptions.

### 📌 Number of Records
* **500 student records**

### 📌 Feature Generation Logic
Each feature was generated using realistic statistical ranges:

| Feature | Distribution Used | Range | Assumption |
| :--- | :--- | :--- | :--- |
| **LMS Login Frequency** | Uniform Integer | 1–30 logins/week | Lower values indicate disengagement |
| **Assignment Delay (avg)** | Uniform Continuous | 0–10 days | Higher delay correlates with stress |
| **Attendance Rate** | Uniform Continuous | 50%–100% | Lower attendance indicates withdrawal |
| **Sentiment Score** | Uniform Continuous | -1 to +1 | Negative values indicate emotional strain |
| **Activity Irregularity** | Uniform Continuous | 0–5 | Higher = unstable engagement behaviour |

## Burnout Risk Construction Logic
A weighted behavioural risk score was engineered using domain assumptions. The underlying logic is represented as:

$$Risk\ Score = (30 - LMS\_Login) \cdot w_1 + Assignment\_Delay \cdot w_2 + (100 - Attendance) \cdot w_3 + (-Sentiment) \cdot w_4 + Irregularity \cdot w_5$$

*(Where weights $w_1$ to $w_5$ simulate behavioural influence strength).*

The final risk score was segmented into three categories:
* Low Risk
* Medium Risk
* High Risk

This approach ensures:
* Structural coherence.
* Predictive separability.
* Behavioural gradient realism.

## Assumptions Embedded in the Dataset
1. Burnout increases with academic delay.
2. Attendance decline precedes severe burnout.
3. Emotional sentiment amplifies behavioural risk.
4. Engagement irregularity signals instability.
5. High-risk cases are rarer than medium-risk cases (class imbalance).

## Limitations of Synthetic Data
* Does not capture true institutional behavioural noise.
* Correlations are rule-based, not organically emergent.
* High-risk class representation may be limited.
* Real-world deployment would require recalibration.

## Why This Is Still Valid for Demonstration
Despite being synthetic, the dataset is highly effective for this project because:
* The dataset exhibits structural coherence.
* Cross-validation stability confirms predictive consistency.
* Behavioural separability mirrors realistic academic patterns.
* It enables a full end-to-end system demonstration (prediction + decision engine + analytics).

## 🧠 Transparency Statement
> This project uses synthetic data strictly for **demonstration purposes, model architecture validation, and system workflow prototyping**. It is designed to be easily adapted to real institutional datasets when available.
