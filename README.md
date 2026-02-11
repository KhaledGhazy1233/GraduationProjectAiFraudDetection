

# 🛡️ AI Behavioral Fraud Detection Terminal (X-100)

An advanced AI-driven security dashboard designed to detect fraudulent banking transactions using **Ensemble Machine Learning** and **Deep Learning**. This project simulates a real-time security terminal that analyzes "Behavioral Fingerprints" to distinguish between legitimate users and sophisticated fraud attempts.


# 🌟 Overview

Unlike traditional fraud detection tools that rely solely on transaction amounts, the **X-100 Terminal** focuses on **Behavioral Analysis**. By processing 28 encrypted behavioral features ( to ) and the transaction amount, the system uses a **Triple-Model Voting Logic** to ensure maximum accuracy and reliability.

# 🚀 Key Features

* **Ensemble Voting System:** Utilizes three distinct AI architectures working in parallel:
* **Random Forest:** For complex decision-tree analysis.
* **XGBoost:** High-performance gradient boosting for precision.
* **ANN (Artificial Neural Network):** Deep learning to capture non-linear fraud patterns.


* **Behavioral Explainer (AI Logic):** The system provides a human-readable "Behavioral Report" for every scan, explaining the *why* behind the decision (e.g., unusual timing, geographic mismatch, or bot-like patterns).
* **Interactive Terminal UI:** A sleek, dark-themed "Control Room" interface featuring real-time scanning animations and dynamic feedback.
* **Randomized Simulation:** Test different financial scenarios against a pool of "Behavioral Fingerprints" to see how the AI adapts.

## 🛠️ Tech Stack

* **Backend:** Python 3.9+, Flask
* **AI/ML Libraries:** TensorFlow/Keras, Scikit-Learn, XGBoost, Pandas, Numpy
* **Frontend:** Modern HTML5, CSS3 (Neon-Dark UI), JavaScript (Async/Await API)
* **Model Management:** Joblib (for ML) & H5 (for Deep Learning)

## 🧠 How It Works (The Logic)

1. **Data Ingestion:** The user inputs an amount. The system randomly selects a behavioral "Object" from a pre-defined JSON dataset.
2. **Preprocessing:** The transaction amount is normalized using a pre-trained `StandardScaler` to match the model's training distribution.
3. **Parallel Inference:** The 29 features are sent to the three models simultaneously.
4. **Majority Voting:**
* If  models flag the transaction, the status is set to **FRAUD**.
* Otherwise, it is marked as **SAFE**.

