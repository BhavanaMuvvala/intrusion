import pandas as pd
import numpy as np
import random

# Parameters
n_samples_per_class = 250  # Balanced number of samples per class
attack_types = ["DoS", "DDoS", "Probe", "U2R", "R2L", "MITM", "SQLi", "Spoofing", "Phishing", "Normal"]
n_samples = n_samples_per_class * len(attack_types)

# Define feature generation logic for each attack type
def generate_data_for_attack(attack_type, n_samples):
    data = {
        "Device_ID": [f"Device_{random.randint(1, 50)}" for _ in range(n_samples)],
        "Packet_Size (bytes)": np.random.randint(64, 1518, n_samples),
        "Latency (ms)": np.random.uniform(1, 500, n_samples).round(2),
        "Throughput (Mbps)": np.random.uniform(0.1, 100, n_samples).round(2),
        "Protocol_Type": np.random.choice(["TCP", "UDP", "ICMP"], n_samples),
        "Source_Port": np.random.randint(1024, 65535, n_samples),
        "Destination_Port": np.random.randint(1, 1024, n_samples),
        "IP_Flag": np.random.choice(["SYN", "ACK", "FIN", "RST", "None"], n_samples),
        "Connection_Duration (ms)": np.random.uniform(10, 10000, n_samples).round(2),
        "Packet_Count": np.random.randint(1, 1000, n_samples),
        "Error_Rate": np.random.uniform(0, 0.1, n_samples).round(4),
        "Fragmentation_Flag": np.random.choice([0, 1], n_samples),
        "Payload_Size (bytes)": np.random.randint(0, 1500, n_samples),
        "Session_Status": np.random.choice(["Established", "Failed", "Pending"], n_samples),
    }

    # Apply specific rules for each attack type
    if attack_type == "DDoS":
        data["Packet_Count"] = np.random.randint(500, 1000, n_samples)
        data["Throughput (Mbps)"] = np.random.uniform(50, 100, n_samples).round(2)
        data["Error_Rate"] = np.random.uniform(0.05, 0.1, n_samples).round(4)
        data["Session_Status"] = np.random.choice(["Failed", "Pending"], n_samples)
    
    elif attack_type == "DoS":
        data["Packet_Count"] = np.random.randint(400, 800, n_samples)
        data["Latency (ms)"] = np.random.uniform(100, 500, n_samples).round(2)
        data["Error_Rate"] = np.random.uniform(0.04, 0.09, n_samples).round(4)
        data["Session_Status"] = "Failed"
    
    elif attack_type == "Probe":
        data["Protocol_Type"] = np.random.choice(["TCP", "ICMP"], n_samples)
        data["Packet_Count"] = np.random.randint(10, 200, n_samples)
        data["Throughput (Mbps)"] = np.random.uniform(0.1, 10, n_samples).round(2)
        data["Session_Status"] = "Established"
    
    elif attack_type == "U2R":
        data["Packet_Size (bytes)"] = np.random.randint(256, 512, n_samples)
        data["Error_Rate"] = np.random.uniform(0.03, 0.08, n_samples).round(4)
        data["Session_Status"] = "Failed"
    
    elif attack_type == "R2L":
        data["Protocol_Type"] = "TCP"
        data["Packet_Count"] = np.random.randint(5, 50, n_samples)
        data["Error_Rate"] = np.random.uniform(0.02, 0.06, n_samples).round(4)
        data["Session_Status"] = "Pending"
    
    elif attack_type == "MITM":
        data["Packet_Size (bytes)"] = np.random.randint(64, 1024, n_samples)
        data["Protocol_Type"] = np.random.choice(["TCP", "UDP"], n_samples)
        data["Latency (ms)"] = np.random.uniform(50, 300, n_samples).round(2)
        data["Error_Rate"] = np.random.uniform(0.01, 0.05, n_samples).round(4)
        data["Session_Status"] = "Pending"
    
    elif attack_type == "SQLi":
        data["Protocol_Type"] = "TCP"
        data["Packet_Size (bytes)"] = np.random.randint(128, 512, n_samples)
        data["Error_Rate"] = np.random.uniform(0.01, 0.04, n_samples).round(4)
        data["Session_Status"] = "Failed"
    
    elif attack_type == "Spoofing":
        data["Source_Port"] = np.random.randint(1, 1024, n_samples)
        data["IP_Flag"] = np.random.choice(["RST", "FIN"], n_samples)
        data["Error_Rate"] = np.random.uniform(0.02, 0.08, n_samples).round(4)
        data["Session_Status"] = np.random.choice(["Failed", "Pending"], n_samples)
    
    elif attack_type == "Phishing":
        data["Packet_Size (bytes)"] = np.random.randint(64, 256, n_samples)
        data["Protocol_Type"] = "TCP"
        data["Error_Rate"] = np.random.uniform(0.01, 0.05, n_samples).round(4)
        data["Session_Status"] = np.random.choice(["Pending", "Failed"], n_samples)
    
    elif attack_type == "Normal":
        data["Packet_Count"] = np.random.randint(1, 500, n_samples)
        data["Throughput (Mbps)"] = np.random.uniform(0.1, 50, n_samples).round(2)
        data["Error_Rate"] = np.random.uniform(0, 0.02, n_samples).round(4)
        data["Session_Status"] = "Established"
    
    data["Attack_Type"] = [attack_type] * n_samples
    return pd.DataFrame(data)

# Combine data for all attack types
df_list = [generate_data_for_attack(attack, n_samples_per_class) for attack in attack_types]
balanced_df = pd.concat(df_list, ignore_index=True)

# Shuffle the data
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
balanced_df.to_csv("multi_balanced_synthetic_network_data.csv", index=False)

print("Balanced synthetic dataset with additional attack types created successfully!")
