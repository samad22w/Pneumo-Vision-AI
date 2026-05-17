import os
# TensorFlow ki extra warnings aur memory optimize karne ke liye
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
from fpdf import FPDF
import gc

# --- 1. MODERN UI & PAGE CONFIG ---
st.set_page_config(page_title="Pneumo-Vision AI", page_icon="🫁", layout="wide")

# Custom CSS for styling (Matching your screenshot)
st.markdown("""
    <style>
    .st-emotion-cache-16txtl3 { padding-top: 1rem; }
    h1, h2, h3 { color: #31333F; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .status-box { background-color: #E8F0FE; padding: 15px; border-radius: 5px; border-left: 5px solid #4285F4; margin-top: 10px; }
    .disclaimer { font-size: 11px; color: #7f8c8d; text-align: center; margin-top: 40px; border-top: 1px solid #eee; padding-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MODEL LOADING & PREPROCESSING ---
@st.cache_resource
def load_pneumonia_model():
    # Make sure your file name matches exactly
    return tf.keras.models.load_model('best_final_model.keras')

model = load_pneumonia_model()

def preprocess_image(image, target_size=(224, 224)):
    img = np.array(image.convert('RGB'))
    img = cv2.resize(img, target_size)
    img_array = np.expand_dims(img, axis=0)
    return img_array.astype('float32') / 255.0

# --- 3. GRAD-CAM LOGIC (FIXED FOR DENSENET) ---
def get_gradcam(img, model):
    try:
        x = preprocess_image(img)
        # Automatically find the last convolutional layer
        target_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D) or 'concat' in layer.name:
                target_layer = layer
                break
        
        grad_model = tf.keras.models.Model([model.inputs], [target_layer.output, model.output])
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x)
            score = predictions[0]

        grads = tape.gradient(score, conv_outputs)
        if grads is None: return np.array(img.convert('RGB'))
            
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        heatmap = heatmap.numpy()
        
        heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
        
        return cv2.addWeighted(np.array(img.convert('RGB')), 0.6, heatmap_img, 0.4, 0)
    except:
        return np.array(img.convert('RGB'))

# --- 4. PDF GENERATION LOGIC ---
def generate_medical_pdf(patient_name, patient_age, gender, ai_diagnosis, confidence, doctor_notes, final_decision, heatmap_path=None):
    pdf = FPDF()
    pdf.add_page()
    
    # Header Section
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(100, 10, "PNEUMO-VISION AI", ln=1)
    
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(100, 5, "Advanced AI Diagnostic Center", ln=0)
    pdf.cell(90, 5, "Developed by: Ab Dus Samad", ln=1, align="R")
    pdf.cell(90, 5, "Bannu, Pakistan", ln=1, align="R")
    
    pdf.line(10, 30, 200, 30)
    pdf.ln(10)
    
    # Patient Data Row
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(100, 8, f"Patient: {patient_name}", ln=0)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(90, 8, f"Reported on: {datetime.now().strftime('%d %b, %Y')}", ln=1, align="R")
    
    pdf.cell(100, 6, f"Age: {patient_age} Years | Sex: {gender}", ln=0)
    pdf.cell(90, 6, "Ref: AI Screening Triage", ln=1, align="R")
    pdf.ln(8)
    
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "X-RAY CHEST ANALYSIS", ln=1, align="C")
    
    # Heatmap Image Insertion
    if heatmap_path:
        pdf.image(heatmap_path, x=65, w=80)
        pdf.ln(5)

    # Findings & Diagnosis
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "AI OBSERVATION:", ln=1)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, "1. Automated visual patterns analyzed via DenseNet121.\n2. Grad-CAM highlights indicate regions of diagnostic interest.")
    
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "IMPRESSION:", ln=1)
    if "PNEUMONIA" in ai_diagnosis:
        pdf.set_text_color(200, 0, 0)
    else:
        pdf.set_text_color(0, 130, 0)
    pdf.cell(0, 8, f"{ai_diagnosis} (Confidence Score: {confidence})", ln=1)
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "DOCTOR'S FINAL VERDICT:", ln=1)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, f"> {final_decision}", ln=1)
    
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "CLINICAL NOTES:", ln=1)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, doctor_notes if doctor_notes else "N/A")

    # Signature Section
    pdf.ln(15)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(60, 10, "Doctor: ", ln=0, align="C")
    pdf.cell(60, 10, "Attending Physician", ln=1, align="C")
    pdf.cell(60, 10, "Signature: ____________________", ln=0, align="C")
    
    
    return bytes(pdf.output())

# --- 5. SIDEBAR NAVIGATION ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3063/3063134.png", width=80)
st.sidebar.title("Pneumo-Vision AI")
st.sidebar.markdown("---")
page = st.sidebar.radio("📋 Navigation Menu", ["🏠 Home", "🩺 Diagnostic Hub", "👨‍💻 About Developer"])
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
    <div class='status-box'>
    <p style='margin-bottom:5px;'>System Status: <b>Online</b> 🟢</p>
    <p style='margin-bottom:0;'>Model: <b>DenseNet121</b></p>
    </div>
    """, unsafe_allow_html=True)

# --- PAGE 1: HOME ---
if page == "🏠 Home":
    st.title("Welcome to Pneumo-Vision AI")
    st.header("Advanced Deep Learning System for Chest Radiography Analysis")
    st.markdown("---")
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.write("Pneumo-Vision AI is a specialized diagnostic assistant designed to rapidly detect Pneumonia from Chest X-Rays using transfer learning.")
        st.subheader("Key Features:")
        st.markdown("*  **Fast Processing:** Analyze images in milliseconds.\n* 🧠 **High Accuracy:** Optimized DenseNet121 architecture.\n* 🔍 **Explainable AI:** Localized heatmaps via Grad-CAM.\n* 📝 **Clinical Reporting:** Professional PDF report generation.")
        st.info("👈 Select 'Diagnostic Hub' to begin analysis.")
    with col2:
        st.image("https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/000001-2.png", use_container_width=True)

# --- PAGE 2: DIAGNOSTIC HUB ---
# --- PAGE 2: DIAGNOSTIC HUB ---
elif page == "🩺 Diagnostic Hub":
    st.title("AI Diagnostic Hub")
    st.write("Upload a patient's Chest X-Ray below to generate an AI diagnosis and heatmap.")
    
    uploaded_file = st.file_uploader("Drop X-Ray file here", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        # Alignment for images
        c_space1, c_img1, c_img2, c_space2 = st.columns([0.1, 1, 1, 0.1])
        
        with st.spinner(" AI is analyzing lung fields..."):
            proc_img = preprocess_image(img)
            prediction = model.predict(proc_img)[0][0]
            heatmap_res = get_gradcam(img, model)
            
        with c_img1:
            st.markdown("<p style='text-align: center; font-weight: bold;'>Original Radiograph</p>", unsafe_allow_html=True)
            st.image(img, use_container_width=True)
        with c_img2:
            st.markdown("<p style='text-align: center; font-weight: bold;'>AI Heatmap (Grad-CAM)</p>", unsafe_allow_html=True)
            st.image(heatmap_res, use_container_width=True)
            
        st.markdown("<br>", unsafe_allow_html=True)

        # --- SENSITIVE THRESHOLD LOGIC (0.3 instead of 0.5) ---
        # Is se Pneumonia detect hone ke chances barh jayenge
        threshold = 0.3 
        is_pneumonia = prediction > threshold
        
        ai_diagnosis = "PNEUMONIA DETECTED" if is_pneumonia else "NORMAL (Healthy Lungs)"
        ai_confidence = f"{(prediction*100):.2f}%" if is_pneumonia else f"{((1-prediction)*100):.2f}%"

        # --- RESTORING THE ALERT BOXES (Your original design) ---
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if is_pneumonia:
                st.error(f"### 🚨 AI Diagnosis: {ai_diagnosis}")
            else:
                st.success(f"### ✅ AI Diagnosis: {ai_diagnosis}")
            st.write(f"**Consensus Confidence Score:** {ai_confidence}")
        
        with res_col2:
            if is_pneumonia:
                st.warning("**Medical Observation:** The AI heatmap highlights areas of high opacity. Red/Yellow zones indicate regions associated with Pneumonia.")
            else:
                st.info("**Medical Observation:** The lung fields appear mostly clear. The AI heatmap highlights the normal structures it verified.")

        st.markdown("---")
        # --- Clinical Report Form & PDF Generation ---
        st.subheader("📝 Clinical Report & Doctor's Approval")
        with st.form("medical_report"):
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1: p_name = st.text_input("Patient Name")
            with f_col2: p_age = st.number_input("Age", 1, 120, 30)
            with f_col3: p_sex = st.selectbox("Gender", ["Male", "Female", "Other"])
            
            p_notes = st.text_area("Clinical Notes")
            p_verdict = st.selectbox("Final Verdict", ["Pneumonia Confirmed", "Normal (Healthy Lungs)", "Requires Further Testing"])
            
            generate_btn = st.form_submit_button("✅ Finalize & Generate PDF Report")
        
        if generate_btn:
            if not p_name:
                st.warning("Please enter patient name.")
            else:
                temp_img = "temp_report_img.png"
                Image.fromarray(heatmap_res).save(temp_img)
                pdf_bytes = generate_medical_pdf(p_name, str(p_age), p_sex, ai_diagnosis, ai_confidence, p_notes, p_verdict, temp_img)
                st.download_button("📥 Download Official Report (PDF)", pdf_bytes, f"Report_{p_name}.pdf", "application/pdf")
                if os.path.exists(temp_img): os.remove(temp_img)
# --- PAGE 3: ABOUT ---
elif page == "👨‍💻 About Developer":
    # --- 1. PROJECT HEADER & MISSION ---
    st.markdown("<h2 style='color: #1e3a8a;'>About Pneumo-Vision AI</h2>", unsafe_allow_html=True)
    
    col_desc, col_tech = st.columns([1.5, 1])
    
    with col_desc:
        st.markdown("""
            ### Mission & Vision
            **Pneumo-Vision AI** is a state-of-the-art deep learning initiative designed to provide 
            rapid and accurate detection of Pneumonia from chest X-ray images. 
            
            Our mission is to bridge the gap in healthcare technology by providing automated 
            diagnostic support to radiologists, especially in underserved and remote regions.
            
            ### Technical Core
            * **Model Architecture:** The system is built on the DenseNet121 architecture, fine-tuned on a large dataset of chest X-rays.
            * **Explainable AI (XAI):** Utilizing Grad-CAM technology, the system visualizes the decision-making process by highlighting specific pathological regions in the lungs.
            
        """)
    
    with col_tech:
        st.markdown("""
            ### Technology Stack
            - **Core Model:** DenseNet121
            - **Framework:** TensorFlow / Keras
            - **Visualization:** Grad-CAM (XAI)
            - **Deployment:** Streamlit
        """)

    st.markdown("---")

    # --- 2. TEAM SECTION STYLING ---
    st.markdown("""
        <style>
        .profile-card {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 100px;
            height: 100%;
        }
        .lead-image {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            object-fit: cover;
            border: 2px solid #3b82f6;
            margin-bottom: 15px;
        }
        .support-image {
            width: 70px;
            height: 70px;
            border-radius: 50%;
            object-fit: cover;
            border: 1px solid #d1d5db;
            margin-bottom: 10px;
        }
        .card-title { color: #1e3a8a; margin: 0; font-size: 18px; }
        .card-subtitle { color: #3b82f6; font-size: 13px; font-weight: 600; text-transform: uppercase; margin-bottom: 8px; }
        .card-text { color: #4b5563; font-size: 13px; line-height: 1.5; }
        .link-text { color: #1e3a8a; font-weight: bold; text-decoration: none; font-size: 12px; }
        </style>
    """, unsafe_allow_html=True)

    # --- 3. LEADERSHIP & SUPPORT TEAM ---
    st.subheader("Development Team")
    
    # Using columns to position your card more naturally rather than fully centered
    lead_col, space_col = st.columns([1.2, 1])
    
    with lead_col:
        st.markdown(f"""
            <div class="profile-card">
                <img src="https://ui-avatars.com/api/?name=Ab+Dus+Samad&background=3b82f6&color=fff&rounded=true&size=150" class="lead-image">
                <div class="card-subtitle">Project Lead</div>
                <h3 class="card-title">Ab Dus Samad</h3>
                <p class="card-text">
                    <b>Software Engineer & AI </b><br>
                    Leading the architectural design and implementation of medical imaging pipelines. 
                    Specialized in DenseNet121 optimization and clinical workflow integration.
                    <br><i>Bannu, Pakistan</i>
                </p>
                <div style="margin-top: 10px;">
                    <a href="https://www.linkedin.com/in/samad-khan-4b3536268/" class="link-text">LinkedIn</a> &nbsp; | &nbsp; 
                    <a href="https://github.com/samad22w" class="link-text">GitHub</a>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Supporting Team Grid
    supp_col1, supp_col2, supp_col3 = st.columns(3)
    
    with supp_col1:
        st.markdown("""
            <div class="profile-card" style="text-align: center;">
                <img src="https://ui-avatars.com/api/?name=Hassan+Durrani&background=0D2B45&color=fff&rounded=true&size=100" class="support-image">
                <h4 class="card-title" style="font-size: 16px;">Hassan Durrani</h4>
                <div class="card-subtitle" style="font-size: 11px;">AI Developer</div>
            </div>
        """, unsafe_allow_html=True)

    with supp_col2:
        st.markdown("""
            <div class="profile-card" style="text-align: center;">
                <img src="https://ui-avatars.com/api/?name=Muskaan&background=0D2B45&color=fff&rounded=true&size=100" class="support-image">
                <h4 class="card-title" style="font-size: 16px;">Muskaan</h4>
                <div class="card-subtitle" style="font-size: 11px;">AI Developer</div>
            </div>
        """, unsafe_allow_html=True)
    with supp_col3:
        st.markdown("""
            <div class="profile-card" style="text-align: center;">
                <img src="https://ui-avatars.com/api/?name=Eng.Nasir+Khan&background=0D2B45&color=fff&rounded=true&size=100" class="support-image">
                <h4 class="card-title" style="font-size: 16px;">Eng. Nasir Khan</h4>
                <div class="card-subtitle" style="font-size: 11px;">Supervisor</div>
            </div>
        """, unsafe_allow_html=True)

st.markdown("<div class='disclaimer'>Disclaimer: This AI tool is for research purposes and educational use only. Clinical correlation by a radiologist is mandatory.</div>", unsafe_allow_html=True)


