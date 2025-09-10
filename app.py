from flask import Flask, render_template, request, jsonify, Response, redirect, url_for
from services.model import predict_message, get_top_spam_keywords, retrain_model
from datetime import datetime
import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

db_config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME")
}

def get_connection():
    return mysql.connector.connect(**db_config)

@app.route("/")
def index():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT message, label, confidence, timestamp FROM predictions ORDER BY id DESC LIMIT 10")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    history = [{"message": row[0], "label": row[1], "confidence": row[2], "timestamp": row[3]} for row in rows]
    return render_template("index.html", history=history)

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]
    pred, confidence = predict_message(message)
    label = "Spam" if pred == 1 else "Ham"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (message, label, confidence, timestamp) VALUES (%s, %s, %s, %s)",
        (message, label, round(confidence, 2), timestamp)
    )
    conn.commit()
    cursor.close()
    conn.close()

    return render_template("index.html", prediction={
        "message": message,
        "label": label,
        "confidence": round(confidence, 2),
        "timestamp": timestamp
    })

@app.route("/export")
def export_csv():
    """Export predictions as CSV directly from DB"""
    def generate():
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT message, label, confidence, timestamp FROM predictions")
        yield "message,label,confidence,timestamp\n"
        for row in cursor.fetchall():
            yield f"\"{row[0]}\",{row[1]},{row[2]},{row[3]}\n"
        cursor.close()
        conn.close()
    return Response(generate(), mimetype="text/csv", headers={"Content-Disposition": "attachment;filename=predictions.csv"})

@app.route("/retrain", methods=["POST"])
def retrain():
    retrain_model("email_data.csv")  # retrain using your dataset
    return redirect(url_for("dashboard"))

@app.route("/dashboard")
def dashboard():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT label, COUNT(*) FROM predictions GROUP BY label")
    counts = dict(cursor.fetchall())
    cursor.close()
    conn.close()
    spam_count = counts.get("Spam", 0)
    ham_count = counts.get("Ham", 0)
    keywords = get_top_spam_keywords()
    return render_template("dashboard.html", spam=spam_count, ham=ham_count, keywords=keywords)

@app.route("/health")
def health():
    return jsonify({"status": "ok", "message": "Spam Classifier is running!"})

if __name__ == "__main__":
    app.run(debug=True)
