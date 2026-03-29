import os
import io
import sqlite3
import requests
import numpy as np
import torch
import torch.nn as nn

from flask import Flask, render_template, request, redirect, session, send_file
from dotenv import load_dotenv
from torchvision import models

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

from preprocessing.image_preprocess import preprocess_image
from preprocessing.clinical_preprocess import preprocess_clinical
from explainability.gradcam import generate_gradcam
from explainability.shap_explainer import explain_clinical

# --------------------------------------------------
# FLASK SETUP
# --------------------------------------------------

app = Flask(__name__)
app.secret_key = "tb_secret_key"

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# CREATE REQUIRED FOLDERS
# --------------------------------------------------

if not os.path.exists("database"):
    os.makedirs("database")

if not os.path.exists("static/uploads"):
    os.makedirs("static/uploads")


# --------------------------------------------------
# DATABASE SETUP
# --------------------------------------------------

def init_db():

    conn = sqlite3.connect("database/tb_system.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        password TEXT,
        role TEXT
    )
    """)

    cursor.execute("""
    INSERT OR IGNORE INTO users(name,email,password,role)
    VALUES('Admin','admin@gmail.com','admin123','admin')
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patient_history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        patient_name TEXT,
        age INTEGER,
        height REAL,
        weight REAL,
        blood_group TEXT,
        cbc REAL,
        esr REAL,
        crp REAL,
        image_path TEXT,
        result TEXT,
        date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

init_db()


# --------------------------------------------------
# LOAD TRAINED CNN MODEL
# --------------------------------------------------

cnn_model = models.resnet18(weights=None)
cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 2)

cnn_model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
cnn_model.to(device)
cnn_model.eval()

print("✅ CNN model loaded successfully")


# --------------------------------------------------
# HOME PAGE → LOGIN
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("login.html")


# --------------------------------------------------
# REGISTER USER
# --------------------------------------------------

@app.route("/register", methods=["GET","POST"])
def register():

    if request.method == "POST":

        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect("database/tb_system.db")
        cur = conn.cursor()

        cur.execute(
        "INSERT INTO users(name,email,password,role) VALUES(?,?,?,?)",
        (name,email,password,"user"))

        conn.commit()
        conn.close()

        return redirect("/")

    return render_template("register.html")


# --------------------------------------------------
# LOGIN
# --------------------------------------------------

@app.route("/login", methods=["POST"])
def login():

    email = request.form["email"]
    password = request.form["password"]

    conn = sqlite3.connect("database/tb_system.db")
    cur = conn.cursor()

    cur.execute(
    "SELECT * FROM users WHERE email=? AND password=?",
    (email,password))

    user = cur.fetchone()

    if user:

        session["user_id"] = user[0]
        session["name"] = user[1]
        session["role"] = user[4]

        if user[4] == "admin":
            return redirect("/admin")

        return redirect("/dashboard")

    return "Invalid Login"


# --------------------------------------------------
# USER DASHBOARD
# --------------------------------------------------

@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:
        return redirect("/")

    return render_template("dashboard.html", name=session["name"])


# --------------------------------------------------
# ADMIN PANEL
# --------------------------------------------------

@app.route("/admin")
def admin():

    conn = sqlite3.connect("database/tb_system.db")
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM users")
    user_count = cur.fetchone()[0]

    cur.execute("""
    SELECT users.name,
           patient_history.patient_name,
           patient_history.age,
           patient_history.cbc,
           patient_history.esr,
           patient_history.crp,
           patient_history.result,
           patient_history.date
    FROM patient_history
    JOIN users
    ON users.id = patient_history.user_id
    """)

    history = cur.fetchall()

    conn.close()

    return render_template(
        "admin.html",
        user_count=user_count,
        history=history
    )


# --------------------------------------------------
# LOGOUT
# --------------------------------------------------

@app.route("/logout")
def logout():

    session.clear()
    return redirect("/")


# --------------------------------------------------
# PREDICT TB
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if "user_id" not in session:
        return redirect("/")

    patient_name = request.form["patient_name"]
    age = request.form["age"]
    height = request.form["height"]
    weight = request.form["weight"]
    blood_group = request.form["blood_group"]

    image_file = request.files["image"]
    image_path = os.path.join("static/uploads", image_file.filename)
    image_file.save(image_path)

    clinical_data = {
        "cbc": float(request.form["cbc"]),
        "esr": float(request.form["esr"]),
        "crp": float(request.form["crp"])
    }

    history = request.form.get("history")

    image = preprocess_image(image_path).unsqueeze(0).to(device)
    clinical = preprocess_clinical(clinical_data).unsqueeze(0).to(device)

    with torch.no_grad():
        tb_out = cnn_model(image)

    tb_pred = torch.argmax(tb_out, dim=1).item()

    tb_label = "TB Detected" if tb_pred == 1 else "Normal"

    cbc = clinical_data["cbc"]
    esr = clinical_data["esr"]
    crp = clinical_data["crp"]

    risk_score = (esr * 0.5) + (crp * 1.5)

    if tb_label == "Normal":
        active_label = "No TB"
    else:
        active_label = "Active TB" if risk_score > 20 else "Latent TB"

    attention_importance = 0.5

    heatmap_path = generate_gradcam(cnn_model, image, clinical, image_path)

    shap_values = explain_clinical(cnn_model, clinical)

    explanation_text = generate_medical_summary(
        tb_label,
        active_label,
        round(risk_score, 2)
    )


    # SAVE HISTORY

    conn = sqlite3.connect("database/tb_system.db")
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO patient_history(
    user_id, patient_name, age, height, weight,
    blood_group, cbc, esr, crp, image_path, result)
    VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """,(
        session["user_id"],
        patient_name,
        age,
        height,
        weight,
        blood_group,
        cbc,
        esr,
        crp,
        image_path,
        tb_label
    ))

    conn.commit()
    conn.close()


    return render_template(
        "result.html",
        tb_status=tb_label,
        tb_type=active_label,
        risk_score=round(risk_score, 2),
        attention_score=round(attention_importance, 3),
        explanation=explanation_text,
        heatmap=heatmap_path
    )


# --------------------------------------------------
# OPENROUTER MEDICAL EXPLANATION
# --------------------------------------------------

def generate_medical_summary(tb_label, active_label, risk_score):

    if not OPENROUTER_API_KEY:
        return "Explanation unavailable."

    prompt = f"""
A diagnostic system produced the following outputs:

Tuberculosis Status: {tb_label}
Type Classification: {active_label}
Risk Score: {risk_score}

Explain this result in simple medical language.
"""

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 200
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        result = response.json()

        return result["choices"][0]["message"]["content"]

    except:
        return "Explanation unavailable."



# --------------------------------------------------
# DOWNLOAD RESULT PDF
# --------------------------------------------------

# --------------------------------------------------
# DOWNLOAD PDF REPORT
# --------------------------------------------------

@app.route("/download_report")
def download_report():

    patient_name = request.args.get("patient_name")
    age = request.args.get("age")
    blood_group = request.args.get("blood_group")

    tb_status = request.args.get("tb_status")
    tb_type = request.args.get("tb_type")

    explanation = request.args.get("explanation")
    heatmap = request.args.get("heatmap")

    buffer = io.BytesIO()

    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("TB Prediction Report", styles["Title"]))
    elements.append(Spacer(1,20))

    # Patient details
    elements.append(Paragraph(f"Patient Name: {patient_name}", styles["Normal"]))
    elements.append(Paragraph(f"Age: {age}", styles["Normal"]))
    elements.append(Paragraph(f"Blood Group: {blood_group}", styles["Normal"]))

    elements.append(Spacer(1,20))

    # Result
    elements.append(Paragraph(f"TB Status: {tb_status}", styles["Normal"]))
    elements.append(Paragraph(f"TB Type: {tb_type}", styles["Normal"]))

    elements.append(Spacer(1,20))

    elements.append(Paragraph("Medical Explanation:", styles["Heading2"]))
    elements.append(Spacer(1,10))
    elements.append(Paragraph(explanation, styles["Normal"]))

    elements.append(Spacer(1,20))

    # GradCAM image
    if heatmap:

        heatmap_path = os.path.join("static", heatmap)

        if os.path.exists(heatmap_path):

            elements.append(Paragraph("GradCAM Visualization:", styles["Heading2"]))
            elements.append(Spacer(1,10))

            gradcam_img = Image(heatmap_path, width=4*inch, height=4*inch)

            elements.append(gradcam_img)

    pdf = SimpleDocTemplate(buffer)
    pdf.build(elements)

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="tb_prediction_report.pdf",
        mimetype="application/pdf"
    )


# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)