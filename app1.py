from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb  # Import XGBoost

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Needed for session management

# Load the trained XGBoost model, scaler, and label encoders
model = xgb.Booster()
model.load_model("xgboost_model.h5")  # Load XGBoost model
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")  # Load saved label encoders

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

# Database setup (SQLite)
DATABASE = "users.db"


def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        device_id TEXT UNIQUE NOT NULL,
        phone TEXT NOT NULL
    )''')
    # Create predictions table
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        device_id TEXT NOT NULL,
        prediction TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()


# Home Page (Login)
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        # Admin login check
        if email == "admin@gmail.com" and password == "admin":
            session["user"] = "admin"  # Store admin in session
            return redirect(url_for("admin"))

        # User login check from database
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):
            session["user"] = user[4]  # Store device_id in session
            return redirect(url_for("predict"))
        else:
            return render_template("login.html", error="Invalid email or password.")
    return render_template("login.html")


# Registration Page
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])
        device_id = request.form["device_id"]
        phone = request.form["phone"]

        try:
            conn = sqlite3.connect(DATABASE)
            c = conn.cursor()
            c.execute("INSERT INTO users (name, email, password, device_id, phone) VALUES (?, ?, ?, ?, ?)",
                      (name, email, password, device_id, phone))
            conn.commit()
            conn.close()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            return render_template("register.html", error="Email or Device ID already exists.")
    return render_template("register.html")


# Prediction Page
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    device_id = session["user"]

    if request.method == "POST":
        user_input = {
            "Device_ID": request.form["Device_ID"],
            "Packet_Size (bytes)": float(request.form["Packet_Size"]),
            "Latency (ms)": float(request.form["Latency"]),
            "Throughput (Mbps)": float(request.form["Throughput"]),
            "Protocol_Type": request.form["Protocol_Type"],
            "Source_Port": int(request.form["Source_Port"]),
            "Destination_Port": int(request.form["Destination_Port"]),
            "IP_Flag": request.form["IP_Flag"],
            "Connection_Duration (ms)": float(request.form["Connection_Duration"]),
            "Packet_Count": int(request.form["Packet_Count"]),
            "Error_Rate": float(request.form["Error_Rate"]),
            "Fragmentation_Flag": int(request.form["Fragmentation_Flag"]),
            "Payload_Size (bytes)": float(request.form["Payload_Size"]),
            "Session_Status": request.form["Session_Status"]
        }

        # Preprocess input
        df_input = pd.DataFrame([user_input])

        # Encode categorical features using the saved label encoders
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

        # Convert input to XGBoost DMatrix
        dtest = xgb.DMatrix(df_input_scaled)

        # Predict and store result
        prediction = model.predict(dtest)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_attack_type = attack_type_mapping[predicted_class]
        import pyttsx3
 
        # init function to get an engine instance for the speech synthesis 
        engine = pyttsx3.init()
        
        # say method on the engine that passing input text to be spoken
        engine.say(predicted_attack_type)
        
        # run and wait method, it processes the voice commands. 
        engine.runAndWait()

        # Store prediction in database
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute("INSERT INTO predictions (device_id, prediction) VALUES (?, ?)",
                  (device_id, predicted_attack_type))
        conn.commit()
        conn.close()

        return render_template("predict.html", device_id=device_id, prediction=predicted_attack_type)

    return render_template("predict.html", device_id=device_id)


# Admin Page
import json

# Load data insights
with open("data_insights.json", "r") as f:
    data_insights = json.load(f)

@app.route("/admin")
def admin():
    if "user" not in session or session["user"] != "admin":
        return redirect(url_for("login"))

    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()

    # Fetch registered users
    c.execute("SELECT * FROM users")
    users = c.fetchall()

    # Fetch prediction history
    c.execute("SELECT * FROM predictions")
    predictions = c.fetchall()

    # Prepare data for graphs
    # 1. Prediction Count Over Time
    c.execute("SELECT DATE(timestamp) as date, COUNT(*) as count FROM predictions GROUP BY date")
    prediction_counts = c.fetchall()

    # 2. Attack Type Prediction Distribution
    c.execute("SELECT prediction, COUNT(*) as count FROM predictions GROUP BY prediction")
    attack_type_predictions = c.fetchall()

    conn.close()

    # Pass data to the template
    return render_template("admin.html", users=users, predictions=predictions,
                          attack_type_distribution=data_insights["attack_type_distribution"],
                          protocol_type_distribution=data_insights["protocol_type_distribution"],
                          prediction_counts=prediction_counts,
                          attack_type_predictions=attack_type_predictions)
# Logout Route
@app.route("/logout")
def logout():
    session.clear()  # Clear all session data
    return redirect(url_for("login"))  # Redirect to login page


if __name__ == "__main__":
    init_db()
    app.run(debug=True)