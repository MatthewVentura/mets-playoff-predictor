# Mets Playoff Predictor: Neural Network Sabermetric Model

This project began as a mathematics symposium presentation during my undergraduate studies.

What started as a school assignment quickly turned into a passion project focused on my favorite team — the New York Mets.

### 🎯 Objective:
Use real sabermetrics and historical MLB data to build a machine learning model that predicts a team’s playoff chances. The spotlight: quantifying the impact of adding Juan Soto to the 2024 Mets roster.

---

## ⚾️ Features Used in the Model (21 total)

- WAR, PA, HR, R, RBI  
- BB%, K%, ISO, AVG, OBP, SLG  
- wOBA, wRC+, OFF, DEF, BSR  
- ERA, WHIP, DefEff, RTOT, Defensive Runs Saved  

---

## 🛠️ Files and What They Do

| File Name                                     | Purpose                                                           |
|----------------------------------------------|-------------------------------------------------------------------|
| `finalSet.py`                                | Trains the neural network using the cleaned dataset               |
| `predict_mets_2024_with_and_without_soto.py` | Simulates Mets playoff chances with/without Juan Soto             |
| `Final_NN_Training_Dataset_2000_2024.csv`    | Cleaned 2000–2024 training dataset (30 teams × 24 seasons)        |
| `Trained_MLB_Model_2000_2024.h5`             | Saved neural network model (Keras `.h5`)                          |
| `final_model_scaler.pkl`                     | StandardScaler object used to normalize input features            |

---

## 🚀 How to Run the Project

### 1️⃣ Train the Neural Network (only once, or after edits)
```bash
python3 finalSet.py
