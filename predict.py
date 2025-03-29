import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import xgboost as xgb

model = xgb.Booster()
model.load_model("xgboost_model.h5")  # Ensure the filename matches the one saved

scaler = joblib.load("scaler.pkl")

# Assuming you have label_encoders saved (for categorical features)
label_encoders = joblib.load("label_encoders.pkl")  # If you saved them earlier

# Attack type mapping (from encoded class values back to original)
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


# Define test input data (user input in a dictionary format)
test_data = [
    {"Device_ID": "Device_45", "Packet_Size (bytes)": 400, "Latency (ms)": 61.33, "Throughput (Mbps)": 17.89,
     "Protocol_Type": "TCP", "Source_Port": 23214, "Destination_Port": 204, "IP_Flag": "SYN",
     "Connection_Duration (ms)": 9216.44, "Packet_Count": 562, "Error_Rate": 0.0305, "Fragmentation_Flag": 0,
     "Payload_Size (bytes)": 905, "Session_Status": "Pending"},
    
    {"Device_ID": "Device_41", "Packet_Size (bytes)": 763, "Latency (ms)": 462.75, "Throughput (Mbps)": 23.18,
     "Protocol_Type": "TCP", "Source_Port": 41247, "Destination_Port": 26, "IP_Flag": "FIN",
     "Connection_Duration (ms)": 9998.76, "Packet_Count": 31, "Error_Rate": 0.0537, "Fragmentation_Flag": 0,
     "Payload_Size (bytes)": 447, "Session_Status": "Pending"},
    
    {"Device_ID": "Device_46", "Packet_Size (bytes)": 900, "Latency (ms)": 335.27, "Throughput (Mbps)": 55.84,
     "Protocol_Type": "TCP", "Source_Port": 61528, "Destination_Port": 513, "IP_Flag": "None",
     "Connection_Duration (ms)": 6094.28, "Packet_Count": 31, "Error_Rate": 0.0343, "Fragmentation_Flag": 0,
     "Payload_Size (bytes)": 336, "Session_Status": "Pending"},
    
    {"Device_ID": "Device_15", "Packet_Size (bytes)": 518, "Latency (ms)": 458.48, "Throughput (Mbps)": 8.89,
     "Protocol_Type": "TCP", "Source_Port": 40468, "Destination_Port": 578, "IP_Flag": "SYN",
     "Connection_Duration (ms)": 9325.57, "Packet_Count": 463, "Error_Rate": 0.0081, "Fragmentation_Flag": 0,
     "Payload_Size (bytes)": 1249, "Session_Status": "Established"},
    
    {"Device_ID": "Device_4", "Packet_Size (bytes)": 282, "Latency (ms)": 137.53, "Throughput (Mbps)": 7.65,
     "Protocol_Type": "TCP", "Source_Port": 61206, "Destination_Port": 669, "IP_Flag": "RST",
     "Connection_Duration (ms)": 1618.19, "Packet_Count": 806, "Error_Rate": 0.0342, "Fragmentation_Flag": 0,
     "Payload_Size (bytes)": 826, "Session_Status": "Failed"},
    
    {"Device_ID": "Device_8", "Packet_Size (bytes)": 187, "Latency (ms)": 51.74, "Throughput (Mbps)": 7.08,
     "Protocol_Type": "ICMP", "Source_Port": 24491, "Destination_Port": 553, "IP_Flag": "ACK",
     "Connection_Duration (ms)": 1850.72, "Packet_Count": 100, "Error_Rate": 0.0791, "Fragmentation_Flag": 0,
     "Payload_Size (bytes)": 667, "Session_Status": "Established"},
    
    {"Device_ID": "Device_30", "Packet_Size (bytes)": 499, "Latency (ms)": 74.84, "Throughput (Mbps)": 68.82,
     "Protocol_Type": "TCP", "Source_Port": 9608, "Destination_Port": 628, "IP_Flag": "RST",
     "Connection_Duration (ms)": 198.98, "Packet_Count": 888, "Error_Rate": 0.0374, "Fragmentation_Flag": 1,
     "Payload_Size (bytes)": 1404, "Session_Status": "Failed"},
    
    {"Device_ID": "Device_39", "Packet_Size (bytes)": 726, "Latency (ms)": 21.48, "Throughput (Mbps)": 26.56,
     "Protocol_Type": "UDP", "Source_Port": 6706, "Destination_Port": 579, "IP_Flag": "FIN",
     "Connection_Duration (ms)": 3974.14, "Packet_Count": 1, "Error_Rate": 0.0084, "Fragmentation_Flag": 0,
     "Payload_Size (bytes)": 807, "Session_Status": "Established"},
    
    {"Device_ID": "Device_41", "Packet_Size (bytes)": 775, "Latency (ms)": 259.35, "Throughput (Mbps)": 69.23,
     "Protocol_Type": "UDP", "Source_Port": 46150, "Destination_Port": 619, "IP_Flag": "FIN",
     "Connection_Duration (ms)": 9522.77, "Packet_Count": 639, "Error_Rate": 0.0846, "Fragmentation_Flag": 0,
     "Payload_Size (bytes)": 199, "Session_Status": "Failed"},
    
    {"Device_ID": "Device_42", "Packet_Size (bytes)": 1174, "Latency (ms)": 378.04, "Throughput (Mbps)": 34.56,
     "Protocol_Type": "UDP", "Source_Port": 13167, "Destination_Port": 479, "IP_Flag": "RST",
     "Connection_Duration (ms)": 6587.63, "Packet_Count": 418, "Error_Rate": 0.009, "Fragmentation_Flag": 1,
     "Payload_Size (bytes)": 329, "Session_Status": "Established"}
]

# Function to preprocess user input
def preprocess_input(user_input):
    # Convert user input into DataFrame
    df_input = pd.DataFrame(user_input)

    # Encode categorical features
    for col in ["Protocol_Type", "IP_Flag", "Session_Status", "Device_ID"]:
        try:
            # Ensure that only known categories are transformed
            df_input[col] = label_encoders[col].transform(df_input[col])
        except ValueError:
            # If unseen category, assign default value (0)
            df_input[col] = 0

    # Select feature columns (same as in training)
    feature_columns = [
        "Device_ID", "Packet_Size (bytes)", "Latency (ms)", "Throughput (Mbps)",
        "Protocol_Type", "Source_Port", "Destination_Port", "IP_Flag",
        "Connection_Duration (ms)", "Packet_Count", "Error_Rate",
        "Fragmentation_Flag", "Payload_Size (bytes)", "Session_Status"
    ]
    df_input = df_input[feature_columns]

    # Scale numerical features using the pre-trained scaler
    df_input_scaled = scaler.transform(df_input)
    return df_input_scaled

# Iterate through each test sample and make predictions
for sample in test_data:
    processed_input = preprocess_input([sample])
    import xgboost as xgb

    dtest = xgb.DMatrix(processed_input)  # Convert input to DMatrix
    prediction = model.predict(dtest)  # Now it works correctly

    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_attack_type = attack_type_mapping[predicted_class]

    # Output the result
    print(f"Input: {sample}")
    print(f"Predicted Attack Type: {predicted_attack_type}")
    print("-" * 50)
