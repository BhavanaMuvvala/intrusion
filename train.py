import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("multi_balanced_synthetic_network_data.csv")

# Encode categorical features and target
categorical_features = ["Device_ID", "Protocol_Type", "IP_Flag", "Session_Status"]
target_feature = "Attack_Type"

# Label encode categorical features
label_encoders = {}
for col in categorical_features + [target_feature]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate features and target
X = df.drop(columns=[target_feature])
y = df[target_feature]

# Scale numerical features for better performance
scaler = StandardScaler()
X = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.pkl")
print("Scaler saved as scaler.pkl")
# Save LabelEncoders
joblib.dump(label_encoders, "label_encoders.pkl")
print("LabelEncoders saved as label_encoders.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# LightGBM Model
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)
lgb_preds = lgb_model.predict(X_test)
lgb_acc = accuracy_score(y_test, lgb_preds)
print(f"LightGBM Accuracy: {lgb_acc:.4f}")

# XGBoost Model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)
print(f"XGBoost Accuracy:np.round {xgb_acc:.4f}")
xgb_model.save_model("xgboost_model.h5")
print("XGBoost model saved as xgboost_model.h5")
# Ensemble (RF + LightGBM + XGBoost)
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
print(f"RandomForest Accuracy: {rf_acc:.4f}")

# Averaging Ensemble
ensemble_preds = (rf_preds + lgb_preds + xgb_preds) / 3
ensemble_preds = np.round(ensemble_preds).astype(int)
ensemble_acc = accuracy_score(y_test, ensemble_preds)
print(f"Ensemble Model Accuracy: {ensemble_acc:.4f}")

# Plot accuracy comparison
models = ["LightGBM", "XGBoost", "RandomForest", "Ensemble"]
accuracies = [lgb_acc, xgb_acc, rf_acc, ensemble_acc]

plt.figure(figsize=(10, 5))
plt.bar(models, accuracies, color=['blue', 'red', 'green', 'purple'])
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.show()
