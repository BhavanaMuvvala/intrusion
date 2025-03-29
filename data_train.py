# training_code.py
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder

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

# 1. Attack Type Distribution
attack_type_counts = df["Attack_Type"].value_counts()

# Map encoded labels back to original attack types
attack_type_mapping = {
    0: "DDoS",
    1: "Normal",
    2: "Phishing",
    3: "Spoofing",
    4: "DoS",
    5: "Probe",
    6: "U2R",
    7: "R2L",
    8: "MITM",
    9: "SQLi"
}

# Convert counts to a dictionary with attack type names
attack_type_distribution = {attack_type_mapping[k]: v for k, v in attack_type_counts.items()}

# 2. Protocol Type Distribution
protocol_type_counts = df["Protocol_Type"].value_counts()

# Map encoded labels back to original protocol types
protocol_type_mapping = {
    0: "TCP",
    1: "UDP",
    2: "ICMP"
}

# Convert counts to a dictionary with protocol type names
protocol_type_distribution = {protocol_type_mapping[k]: v for k, v in protocol_type_counts.items()}

# 3. Save these distributions to a file (e.g., JSON)
with open("data_insights.json", "w") as f:
    json.dump({
        "attack_type_distribution": attack_type_distribution,
        "protocol_type_distribution": protocol_type_distribution
    }, f)

print("Data insights saved to data_insights.json")