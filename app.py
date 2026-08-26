import os
import re
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="Focus & Concentration Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Minimalistic Bold Aesthetics
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        color: #F8FAFC;
        padding: 2rem 2.5rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    .header-card h1 {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        margin-bottom: 0.5rem;
        font-size: 2.2rem;
    }
    .header-card p {
        color: #94A3B8;
        font-size: 1.05rem;
        font-weight: 500;
        margin: 0;
    }

    /* Force all selectbox labels to be bold cleanly */
    .stSelectbox label {
        font-weight: 700 !important;
        font-size: 1rem !important;
        color: #1E293B !important;
    }

    /* Result Cards */
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    
    /* Custom Badge Styles */
    .badge-5 { background-color: #DCFCE7; color: #15803D; border: 2px solid #86EFAC; }
    .badge-4 { background-color: #E0F2FE; color: #0369A1; border: 2px solid #7DD3FC; }
    .badge-3 { background-color: #FEF9C3; color: #A16207; border: 2px solid #FDE047; }
    .badge-2 { background-color: #FFEDD5; color: #C2410C; border: 2px solid #FDBA74; }
    .badge-1 { background-color: #FEE2E2; color: #B91C1C; border: 2px solid #FCA5A5; }
    
    .score-badge {
        display: inline-block;
        padding: 0.6rem 1.8rem;
        font-size: 2rem;
        font-weight: 900 !important;
        border-radius: 50px;
        margin: 0.8rem 0;
    }
    
    /* Recommendation Item */
    .rec-item {
        background: #FFFFFF;
        border-left: 4px solid #3B82F6;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.7rem;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        font-size: 0.98rem;
    }
    .rec-item b {
        color: #0F172A;
        font-weight: 700;
    }
    
    /* Button styling */
    div.stButton > button {
        font-weight: 800 !important;
        font-size: 1.05rem !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

MODEL_FILE = r"model.pkl"
if not os.path.exists(MODEL_FILE):
    MODEL_FILE = r"c:\Users\prath\Downloads\ML Project\model.pkl"

def clean_str(val):
    if not isinstance(val, str):
        return val
    return re.sub(r'[\x96\u2013\u2014â€“â€”]+', '-', val).strip()

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_FILE):
        import train_model
        train_model.main()
    with open(MODEL_FILE, 'rb') as f:
        return pickle.load(f)

try:
    artifacts = load_artifacts()
    model = artifacts['model']
    encoders = artifacts['encoders']
    feature_cols = artifacts['feature_cols']
    feature_options = artifacts['feature_options']
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# Header Section
st.markdown("""
<div class="header-card">
    <h1>🎯 Concentration & Focus Predictor</h1>
    <p>Predict your daily focus level based on work habits, digital distractions, and sleep patterns.</p>
</div>
""", unsafe_allow_html=True)

# User Input Form
st.markdown("### 📋 Enter Your Daily Habits & Environment")

# Mapping technical column names to user-friendly titles
labels = {
    'Age Group': 'Age Group',
    'Gender': 'Gender',
    'Role': 'Role / Occupation',
    '1.Average daily screen time (in hours)?': '1. Daily Screen Time',
    '2.Which type of digital distraction affects you most while studying/working?': '2. Primary Digital Distraction',
    '3.How often do you check your phone during a study/work session': '3. Phone Checking Frequency',
    '4.Do notifications interrupt your focus?': '4. Notification Interruptions',
    '5.Average duration of a focused study/work session (without breaks)?': '5. Focused Session Duration',
    '6.Do you multitask while studying/working (music, chats, tabs open)?': '6. Multitasking Frequency',
    '7.At which time do you feel MOST distracted?': '7. Peak Distraction Time',
    '8.How many breaks do you usually take in one hour?': '8. Hourly Break Count',
    '9.Average sleep duration per day?': '9. Daily Sleep Duration',
    '10.Do you use your phone just before sleeping?': '10. Phone Use Before Sleep',
    '12.How refreshed do you feel after waking up?': '12. Morning Refreshment Level',
    '13.After getting distracted, how long does it usually take you to regain full focus?': '13. Time to Regain Focus'
}

col1, col2 = st.columns(2)

input_data = {}

# Split columns evenly across two form panels
keys = list(feature_cols)
half = (len(keys) + 1) // 2
left_keys = keys[:half]
right_keys = keys[half:]

with col1:
    st.markdown("#### 👤 Profile & Schedule")
    for key in left_keys:
        raw_opts = feature_options.get(key, [])
        opts = [clean_str(o) for o in raw_opts]
        label = labels.get(key, key)
        input_data[key] = st.selectbox(label, options=opts, key=key)

with col2:
    st.markdown("#### 📱 Distractions & Recovery")
    for key in right_keys:
        raw_opts = feature_options.get(key, [])
        opts = [clean_str(o) for o in raw_opts]
        label = labels.get(key, key)
        input_data[key] = st.selectbox(label, options=opts, key=key)

st.markdown("---")

predict_col, _ = st.columns([1, 2])
with predict_col:
    submit_button = st.button("🚀 Predict Concentration Score", type="primary", use_container_width=True)

if submit_button:
    # Prepare encoded input array matching feature order
    encoded_input = []
    for col in feature_cols:
        val = str(input_data[col])
        encoder = encoders[col]
        # Handle unseen values safely
        if val in encoder.classes_:
            enc_val = encoder.transform([val])[0]
        else:
            enc_val = 0
        encoded_input.append(enc_val)
    
    input_df = pd.DataFrame([encoded_input], columns=feature_cols)
    prediction = model.predict(input_df)[0]
    
    # Interpretation text and badge styling
    badge_class = f"badge-{prediction}"
    rating_desc = {
        5: ("Excellent Focus (5 / 5)", "🎉 <b>Outstanding!</b> You have optimal focus habits and minimal digital friction."),
        4: ("Good Focus (4 / 5)", "👍 <b>Great concentration.</b> Minor tweaks can make your focus even sharper."),
        3: ("Moderate Focus (3 / 5)", "⚖️ <b>Fair concentration.</b> Digital distractions or sleep habits may be affecting your momentum."),
        2: ("Low Focus (2 / 5)", "⚠️ <b>Focus needs attention.</b> Frequent phone checks or notifications are breaking your flow."),
        1: ("Very Low Focus (1 / 5)", "🚨 <b>High Distraction Risk.</b> Immediate changes to work environment and phone usage recommended.")
    }
    
    title, desc = rating_desc.get(prediction, ("Concentration Rating", ""))

    st.markdown("### 📊 Prediction Result")
    
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.markdown(f"""
        <div style="text-align: center; padding: 1.5rem; background: #FFFFFF; border-radius: 12px; border: 1px solid #E2E8F0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
            <p style="margin: 0; color: #64748B; font-weight: 700; text-transform: uppercase; font-size: 0.85rem;">PREDICTED CONCENTRATION LEVEL</p>
            <div class="score-badge {badge_class}">{prediction} / 5</div>
            <p style="font-weight: 800; font-size: 1.1rem; margin-top: 0.5rem; color: #1E293B;">{title}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with res_col2:
        st.info(desc)
        
        # Personalized Actionable Recommendations
        st.markdown("##### 💡 Actionable Productivity Tips:")
        
        tips = []
        if "Very frequently" in input_data.get('3.How often do you check your phone during a study/work session', '') or "15" in input_data.get('3.How often do you check your phone during a study/work session', ''):
            tips.append("📱 <b>Reduce Phone Checking</b>: Use techniques like the Pomodoro method (25m focus / 5m rest) and place your phone in another room.")
            
        if input_data.get('4.Do notifications interrupt your focus?', '') in ['Often', 'Always']:
            tips.append("🔕 <b>Silence Notifications</b>: Turn on 'Do Not Disturb' or Focus mode while working to avoid context switching.")
            
        if input_data.get('10.Do you use your phone just before sleeping?', '') in ['Always', 'Often']:
            tips.append("🌙 <b>Nighttime Wind-down</b>: Avoid screens 30-45 minutes before sleep to improve deep sleep quality.")
            
        if input_data.get('6.Do you multitask while studying/working (music, chats, tabs open)?', '') in ['Often', 'Always']:
            tips.append("🎯 <b>Single-Tasking</b>: Close unnecessary browser tabs and communication apps during deep work sessions.")
            
        if not tips:
            tips.append("✨ <b>Great Routine!</b> Maintain your current habits and keep tracking your session durations.")
            
        for tip in tips:
            st.markdown(f"<div class='rec-item'>{tip}</div>", unsafe_allow_html=True)

# Feature Importance Section
st.markdown("---")
with st.expander("📈 View Feature Importance Analysis"):
    st.markdown("See which factors have the highest overall influence on concentration predictions in this model:")
    
    importances = model.feature_importances_
    clean_feature_names = [labels.get(col, col) for col in feature_cols]
    
    feat_df = pd.DataFrame({
        'Feature': clean_feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(feat_df['Feature'], feat_df['Importance'], color='#3B82F6')
    ax.set_xlabel('Relative Importance', fontweight='bold')
    ax.set_title('Factors Impacting Concentration Ability', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
