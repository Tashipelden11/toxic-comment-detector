
import streamlit as st
import joblib
import re

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Toxic Comment Detector",
    page_icon="🛡️",
    layout="centered"
)

# --------------------------------------------------
# LOAD MODEL AND VECTORIZER
# --------------------------------------------------

model = joblib.load("toxic_svm_model.pkl")
vectorizer = joblib.load("toxic_tfidf_vectorizer.pkl")


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background: #F4F7FB;
}

.block-container {
    max-width: 900px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}


/* ---------- HEADER ---------- */

.hero {
    background: linear-gradient(135deg, #172554, #3730A3);
    padding: 38px 30px;
    border-radius: 22px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 8px 25px rgba(30, 41, 59, 0.12);
}

.hero h1 {
    color: white;
    font-size: 40px;
    margin-bottom: 8px;
}

.hero p {
    color: #DCE4FF;
    font-size: 16px;
    margin-bottom: 0;
}


/* ---------- SECTION TITLES ---------- */

.section-title {
    color: #172554;
    font-size: 21px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 12px;
}


/* ---------- INPUT ---------- */

.stTextArea textarea {
    background-color: white;
    color: #1E293B;
    border: 1px solid #CBD5E1;
    border-radius: 14px;
    font-size: 15px;
    padding: 14px;
}

.stTextArea textarea:focus {
    border: 2px solid #6366F1;
}


/* ---------- BUTTON ---------- */

.stButton > button {
    background: linear-gradient(135deg, #3730A3, #4F46E5);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 11px;
    font-size: 16px;
    font-weight: 600;
    box-shadow: 0 5px 15px rgba(79, 70, 229, 0.20);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #312E81, #4338CA);
}


/* ---------- RESULT ---------- */

.result {
    padding: 24px;
    border-radius: 16px;
    text-align: center;
    margin-top: 20px;
    font-size: 24px;
    font-weight: 700;
}


/* ---------- METRICS ---------- */

.metric-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 15px;
    padding: 18px 10px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(30, 41, 59, 0.05);
}

.metric-value {
    color: #3730A3;
    font-size: 25px;
    font-weight: 700;
}

.metric-label {
    color: #64748B;
    font-size: 13px;
    margin-top: 4px;
}


/* ---------- WORKFLOW ---------- */

.workflow {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 22px;
    text-align: center;
    color: #475569;
    box-shadow: 0 4px 15px rgba(30, 41, 59, 0.04);
}

.workflow span {
    color: #4F46E5;
    font-weight: 700;
}


/* ---------- FOOTER ---------- */

.footer {
    text-align: center;
    color: #94A3B8;
    font-size: 13px;
    margin-top: 35px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">

<h1>🛡️ Toxic Comment Detector</h1>

<p>
Classical NLP application using TF-IDF and Linear SVM
to identify potentially toxic comments.
</p>

</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# COMMENT INPUT
# --------------------------------------------------

st.markdown(
    '<div class="section-title">💬 Analyze a Comment</div>',
    unsafe_allow_html=True
)

comment = st.text_area(
    "Comment",
    placeholder="Type or paste a comment here...",
    height=170,
    label_visibility="collapsed"
)

analyze = st.button(
    "🔍 Analyze Comment",
    use_container_width=True
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if analyze:

    if not comment.strip():

        st.warning("Please enter a comment before predicting.")

    else:

        # Clean the input text
        cleaned_comment = clean_text(comment)

        # Convert text into TF-IDF features
        comment_vector = vectorizer.transform([cleaned_comment])

        # Predict using the trained SVM model
        prediction = model.predict(comment_vector)[0]

        if prediction == 1:

            st.markdown("""
            <div class="result"
            style="
            background:#FFF1F2;
            color:#BE123C;
            border:1px solid #FDA4AF;">
            🚨 Prediction: Toxic
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="result"
            style="
            background:#ECFDF5;
            color:#047857;
            border:1px solid #86EFAC;">
            ✓ Prediction: Non-toxic
            </div>
            """, unsafe_allow_html=True)


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📊 Model Performance</div>',
    unsafe_allow_html=True
)

st.caption("Performance on the held-out test set.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">96.12%</div>
        <div class="metric-label">Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">86.29%</div>
        <div class="metric-label">Precision</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">70.77%</div>
        <div class="metric-label">Recall</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">77.77%</div>
        <div class="metric-label">F1-score</div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# WORKFLOW
# --------------------------------------------------

st.markdown(
    '<div class="section-title">⚙️ How It Works</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="workflow">

<span>Input</span>
&nbsp; → &nbsp;
<span>Cleaning</span>
&nbsp; → &nbsp;
<span>TF-IDF</span>
&nbsp; → &nbsp;
<span>Linear SVM</span>
&nbsp; → &nbsp;
<span>Prediction</span>

</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">

<hr>

Classical NLP • TF-IDF • Linear SVM

</div>
""", unsafe_allow_html=True)
