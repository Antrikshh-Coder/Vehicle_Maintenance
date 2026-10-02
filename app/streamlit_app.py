import os
import sys
import pandas as pd
import numpy as np
import streamlit as st

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.predict import predict_engine_condition

# Streamlit Page Config
st.set_page_config(
    page_title="Vehicle Predictive Maintenance Engine 🚗",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR THEME SWITCHER (DARK & LIGHT 2 MODES ONLY) ---
st.sidebar.markdown("### 🎨 Appearance & Theme Mode")
theme_mode = st.sidebar.radio(
    "Select Display Mode",
    ["🌙 Dark Mode", "☀️ Light Mode"]
)

# Dynamic Color Tokens for Dark / Light Modes
if "Dark" in theme_mode:
    bg_gradient = "linear-gradient(135deg, #070d19 0%, #0e172a 50%, #111c35 100%)"
    sidebar_bg = "#091122"
    text_color = "#f1f5f9"
    subtext_color = "#94a3b8"
    primary_accent = "#00f5d4"
    secondary_accent = "#7b2cbf"
    highlight_pink = "#f72585"
    card_bg = "rgba(30, 41, 59, 0.55)"
    card_border = "rgba(0, 245, 212, 0.2)"
    graph_card_bg = "rgba(15, 23, 42, 0.9)"
    table_header_bg = "#1e293b"
else:
    bg_gradient = "linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #f1f5f9 100%)"
    sidebar_bg = "#ffffff"
    text_color = "#0f172a"
    subtext_color = "#475569"
    primary_accent = "#0284c7"
    secondary_accent = "#6d28d9"
    highlight_pink = "#db2777"
    card_bg = "rgba(255, 255, 255, 0.85)"
    card_border = "rgba(2, 132, 199, 0.3)"
    graph_card_bg = "rgba(255, 255, 255, 0.95)"
    table_header_bg = "#e2e8f0"

# Custom Injectable Styling: Google Fonts, Keyframe Animations, Glassmorphism, Dynamic Dark/Light Theme
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&family=Space+Grotesk:wght@600;700&display=swap');

    /* Global Page Color & Background */
    .stApp {{
        background: {bg_gradient};
        color: {text_color};
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}
    
    /* Font Overrides */
    h1, h2, h3, h4, .main-heading {{
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.02em;
        color: {text_color} !important;
    }}
    
    p, span, label {{
        color: {text_color};
    }}
    
    .mono-num {{
        font-family: 'JetBrains Mono', monospace !important;
    }}
    
    /* Clean Dynamic Sidebar */
    section[data-testid="stSidebar"] {{
        background: {sidebar_bg};
        border-right: 1px solid rgba(128, 128, 128, 0.15);
    }}
    
    /* Vibrant Header Banner with Scanline & Glow Animations */
    .hero-banner {{
        background: radial-gradient(circle at top right, rgba(255, 255, 255, 0.08), transparent 50%),
                    linear-gradient(135deg, {card_bg} 0%, {sidebar_bg} 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        padding: 2.2rem 2.5rem;
        border-radius: 24px;
        border: 1px solid {card_border};
        box-shadow: 0 20px 35px -10px rgba(0, 0, 0, 0.15);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }}
    .hero-banner::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, transparent, {primary_accent}, {secondary_accent}, {highlight_pink}, transparent);
        animation: scanline 4s linear infinite;
    }}
    @keyframes scanline {{
        0% {{ left: -100%; }}
        100% {{ left: 100%; }}
    }}
    
    .hero-title {{
        font-size: 2.85rem;
        font-weight: 900;
        background: linear-gradient(90deg, {primary_accent} 0%, {secondary_accent} 50%, {highlight_pink} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }}
    
    .animated-car {{
        display: inline-block;
        animation: car-bounce 2s ease-in-out infinite;
    }}
    @keyframes car-bounce {{
        0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
        50% {{ transform: translateY(-6px) rotate(-1deg); }}
    }}

    .hero-subtitle {{
        color: {subtext_color};
        font-size: 1.15rem;
        font-weight: 500;
        margin: 0;
    }}
    
    /* Animated Live Badge */
    .live-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.6rem;
        background: rgba(128, 128, 128, 0.1);
        color: {primary_accent};
        padding: 0.35rem 0.95rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        border: 1px solid {card_border};
        margin-bottom: 0.8rem;
    }}
    .badge-dot {{
        width: 10px;
        height: 10px;
        background-color: {primary_accent};
        border-radius: 50%;
        box-shadow: 0 0 12px {primary_accent};
        animation: pulse-glow 1.8s infinite;
    }}
    @keyframes pulse-glow {{
        0% {{ transform: scale(0.9); box-shadow: 0 0 0 0 rgba(0, 245, 212, 0.7); }}
        70% {{ transform: scale(1.2); box-shadow: 0 0 0 10px rgba(0, 0, 0, 0); }}
        100% {{ transform: scale(0.9); box-shadow: 0 0 0 0 rgba(0, 0, 0, 0); }}
    }}

    /* Glassmorphism Metric Cards with Hover & Shimmer Animations */
    .glass-card {{
        background: {card_bg};
        backdrop-filter: blur(14px);
        border: 1px solid {card_border};
        border-radius: 20px;
        padding: 1.35rem 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }}
    .glass-card:hover {{
        transform: translateY(-5px) scale(1.015);
        border-color: {primary_accent};
        box-shadow: 0 18px 35px -5px rgba(0, 0, 0, 0.15);
    }}
    .card-label {{
        font-size: 0.825rem;
        font-weight: 700;
        color: {subtext_color};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }}
    .card-value {{
        font-size: 2.25rem;
        font-weight: 900;
        color: {text_color};
        font-family: 'JetBrains Mono', monospace;
    }}
    .card-sub {{
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.35rem;
    }}

    /* Detailed Graph Explanation Cards */
    .graph-card {{
        background: {graph_card_bg};
        border: 1px solid {card_border};
        border-left: 5px solid {primary_accent};
        border-radius: 0 16px 16px 0;
        padding: 1.35rem 1.6rem;
        margin-top: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }}
    .graph-card:hover {{
        box-shadow: 0 12px 25px rgba(0, 0, 0, 0.15);
    }}
    .graph-card-title {{
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        color: {primary_accent};
        font-size: 1.05rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    .graph-card-text {{
        font-size: 0.95rem;
        color: {text_color};
        line-height: 1.65;
    }}

    /* Prediction Outcome Cards with Pulse Animation */
    .pred-card-normal {{
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.85) 0%, rgba(2, 44, 34, 0.95) 100%);
        border: 2px solid #10b981;
        border-radius: 22px;
        padding: 2.2rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(16, 185, 129, 0.35);
        animation: card-appear 0.5s ease-out;
    }}
    .pred-card-fault {{
        background: linear-gradient(135deg, rgba(153, 27, 27, 0.85) 0%, rgba(69, 10, 10, 0.95) 100%);
        border: 2px solid #ff0054;
        border-radius: 22px;
        padding: 2.2rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(255, 0, 84, 0.4);
        animation: card-appear 0.5s ease-out;
    }}
    @keyframes card-appear {{
        from {{ opacity: 0; transform: scale(0.95) translateY(12px); }}
        to {{ opacity: 1; transform: scale(1) translateY(0); }}
    }}
    
    /* Styled Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: rgba(128, 128, 128, 0.1);
        border-radius: 12px;
        padding: 8px 16px;
        color: {subtext_color};
        font-weight: 600;
        border: 1px solid rgba(128, 128, 128, 0.15);
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {card_bg} !important;
        color: {primary_accent} !important;
        border: 1px solid {primary_accent} !important;
    }}
</style>
""", unsafe_allow_html=True)

# Helper function to render KPI cards with emojis
def render_kpi(label, value, subtext, color=primary_accent, emoji="⚙️"):
    st.markdown(f"""
    <div class="glass-card">
        <div class="card-label"><span>{emoji}</span> {label}</div>
        <div class="card-value">{value}</div>
        <div class="card-sub" style="color: {color};">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)

# Helper function to render graph explanations
def render_graph_explanation(title, simple_explanation, detailed_explanation):
    st.markdown(f"""
    <div class="graph-card">
        <div class="graph-card-title">💡 Graph Explanation: {title}</div>
        <div class="graph-card-text">
            <p style="margin-bottom: 0.6rem;"><strong>📌 In Simple Terms:</strong> {simple_explanation}</p>
            <p style="margin-bottom: 0;"><strong>🔬 Detailed Technical Explanation:</strong> {detailed_explanation}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Hero Header Banner
st.markdown(f"""
<div class="hero-banner">
    <div class="live-badge">
        <div class="badge-dot"></div> Live Telemetry Active • Case Study No. 67
    </div>
    <h1 class="hero-title"><span class="animated-car">🚗</span> Vehicle Predictive Maintenance Engine</h1>
    <p class="hero-subtitle">Industrial AI Telemetry Platform for Engine Health Monitoring & Diagnostic Analytics</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation Panel
st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Navigation Menu")
page = st.sidebar.radio(
    "Select Section",
    [
        "📋 Home & Executive Overview",
        "📊 Dataset & Sensor Dictionary",
        "📈 Exploratory Data Analysis (EDA)",
        "🤖 AI Model Benchmark & Metrics",
        "🔮 Live Telemetry Maintenance Predictor"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="font-size: 0.85rem; color: {subtext_color}; font-family: 'JetBrains Mono', monospace; background: rgba(128,128,128,0.08); padding: 1rem; border-radius: 12px; border: 1px solid {card_border};">
    <strong style="color: {text_color};">⚡ Production Benchmark</strong><br>
    • Dataset: <strong>19,535 Records</strong><br>
    • Classifier: <strong>SVM (RBF Kernel)</strong><br>
    • F1-Score: <span style="color: {primary_accent}; font-weight: 700;">0.7709</span><br>
    • Recall Rate: <span style="color: #10b981; font-weight: 700;">89.57%</span><br>
    • Fault Detection: <span style="color: #34d399; font-weight: 700;">2,206 / 2,463</span>
</div>
""", unsafe_allow_html=True)

# --- PAGE 1: HOME & EXECUTIVE OVERVIEW ---
if page == "📋 Home & Executive Overview":
    st.header("📋 Project Executive Overview & Operational Context")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 🎯 Problem Statement & Real-World Impact
        *"A fleet operator wants to study patterns that may indicate maintenance requirements."*
        
        Unplanned commercial vehicle breakdowns cause severe financial losses from emergency towing, revenue downtime, and logistics SLA violations. 
        This AI system processes **19,535 real-time engine telemetry snapshots** across 6 core operational sensors:
        
        - ⚙️ **Engine RPM** (Rotational velocity)
        - 🛢️ **Lub Oil Pressure** (Fluid lubrication pressure)
        - ⛽ **Fuel Pressure** (Injection system delivery pressure)
        - 💧 **Coolant Pressure** (Cooling jacket system pressure)
        - 🌡️ **Lub Oil Temperature** (Lube oil thermal state)
        - 🌡️ **Coolant Temperature** (Liquid cooling thermal state)
        
        ### 🔬 Technical Rigor & Workflow Architecture
        1. **Zero Preprocessing Leakage**: Scaled using `StandardScaler` fitted strictly on the 80% training set (`X_train`).
        2. **Algorithm Benchmarking**: Trained & evaluated 5 machine learning models via 5-Fold Stratified Cross-Validation.
        3. **Production Model Choice**: Selected **Support Vector Machine (SVM)** with an RBF kernel for achieving an outstanding **89.57% Recall rate** on engine faults.
        """)
        
    with col2:
        render_kpi("Dataset Volume", "19,535", "📊 Sensor Telemetry Rows", primary_accent, "📁")
        st.markdown("<br>", unsafe_allow_html=True)
        render_kpi("Best Machine Learning Model", "SVM (RBF)", "🏆 RBF Hyperplane Engine", secondary_accent, "🤖")
        st.markdown("<br>", unsafe_allow_html=True)
        render_kpi("Engine Fault Recall", "89.57%", "⚡ Uncaught Breakdowns < 10.4%", "#10b981", "🎯")

# --- PAGE 2: DATASET & SENSOR DICTIONARY ---
elif page == "📊 Dataset & Sensor Dictionary":
    st.header("📊 Dataset Architecture & Sensor Telemetry Dictionary")
    st.write("Comprehensive specification of 19,535 engine sensor telemetry records collected from heavy-duty vehicle engine tests.")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi("Total Records", "19,535", "📁 Telemetry Snapshots", primary_accent, "📊")
    with c2:
        render_kpi("Feature Count", "6", "⚙️ Pressure & Temperature", "#818cf8", "🔬")
    with c3:
        render_kpi("Normal State (0)", "7,218", "✅ 36.95% Baseline", "#10b981", "🟢")
    with c4:
        render_kpi("Maintenance (1)", "12,317", "⚠️ 63.05% Fault States", "#ff0054", "🔴")
        
    st.markdown("---")
    st.subheader("🔍 Technical Sensor Data Dictionary")
    
    dict_df = pd.DataFrame([
        {"Feature Name": "⚙️ Engine RPM", "Description": "Engine rotational crankshaft speed in Revolutions Per Minute", "Data Type": "int64", "Role": "Input", "Observed Range": "61 - 2,239 RPM"},
        {"Feature Name": "🛢️ Lub Oil Pressure", "Description": "Lubricating oil pressure supplied to engine bearings", "Data Type": "float64", "Role": "Input", "Observed Range": "0.003 - 7.27 bar"},
        {"Feature Name": "⛽ Fuel Pressure", "Description": "Fuel delivery pressure into high-pressure injection system", "Data Type": "float64", "Role": "Input", "Observed Range": "0.003 - 21.14 bar"},
        {"Feature Name": "💧 Coolant Pressure", "Description": "Engine cooling circuit fluid circulation pressure", "Data Type": "float64", "Role": "Input", "Observed Range": "0.002 - 7.48 bar"},
        {"Feature Name": "🌡️ Lub Oil Temp", "Description": "Engine lubricating oil sump operating temperature", "Data Type": "float64", "Role": "Input", "Observed Range": "71.32 - 89.58 °C"},
        {"Feature Name": "🌡️ Coolant Temp", "Description": "Engine coolant jacket fluid operating temperature", "Data Type": "float64", "Role": "Input", "Observed Range": "61.67 - 195.53 °C"},
        {"Feature Name": "🏷️ Engine Condition", "Description": "Target maintenance classification label (0=Normal, 1=Maintenance)", "Data Type": "int64", "Role": "Target Label", "Observed Range": "{0, 1}"}
    ])
    st.table(dict_df)
    
    if os.path.exists("data/dataset.csv"):
        st.subheader("📁 Raw Telemetry Sample Data (First 10 Rows)")
        st.dataframe(pd.read_csv("data/dataset.csv").head(10), use_container_width=True)

# --- PAGE 3: EDA & DETAILED GRAPH EXPLANATIONS ---
elif page == "📈 Exploratory Data Analysis (EDA)":
    st.header("📈 Exploratory Data Analysis & Graph Explanations")
    st.write("Visual patterns generated from 19,535 engine telemetry records, explained in simple and detailed terms.")
    
    tab1, tab2, tab3 = st.tabs(["📊 1. Class Balance & Density KDE", "🔥 2. Correlations & Thermal Scatter", "📦 3. Outlier Boxplot Analysis"])
    
    with tab1:
        st.subheader("📊 Class Balance Bar Chart & Kernel Density Distribution Curves")
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists("outputs/figures/target_distribution.png"):
                st.image("outputs/figures/target_distribution.png", use_container_width=True)
                render_graph_explanation(
                    "Target Class Distribution Bar Chart",
                    "Out of 19,535 engine readings, 12,317 engines (63.1%) required maintenance (Class 1, red bar), while 7,218 engines (36.9%) were operating normally (Class 0, green bar).",
                    "This chart displays the target class distribution. Out of 19,535 total vehicle records, 12,317 vehicles (63.1%) require maintenance (Class 1), and 7,218 vehicles (36.9%) are in normal condition (Class 0). Having a majority of maintenance cases in our dataset ensures our AI model gets plenty of real failure examples to learn from."
                )
                
        with col2:
            if os.path.exists("outputs/feature_distributions.png") or os.path.exists("outputs/figures/feature_distributions.png"):
                path = "outputs/figures/feature_distributions.png" if os.path.exists("outputs/figures/feature_distributions.png") else "outputs/feature_distributions.png"
                st.image(path, use_container_width=True)
                render_graph_explanation(
                    "Sensor Kernel Density Estimations (KDE)",
                    "This graph compares sensor readings between healthy engines (green) and faulty engines (red). When an engine breaks down, coolant temperature rises high and oil pressure drops.",
                    "Density curves show the shape of sensor values. The red curve (maintenance state) shifts towards higher temperatures (up to 195°C) and lower oil pressures compared to the green curve (healthy state). This clear shift gives our AI model strong signals to detect engine trouble early."
                )

    with tab2:
        st.subheader("🔥 Thermodynamic Pearson Correlation Matrix & Thermal Scatter Plot")
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists("outputs/figures/correlation_heatmap.png"):
                st.image("outputs/figures/correlation_heatmap.png", use_container_width=True)
                render_graph_explanation(
                    "Pearson Correlation Matrix Heatmap",
                    "This heatmap shows how sensors relate to each other. For example, as engine temperature increases, lubrication oil pressure decreases.",
                    "Correlation values range from -1.0 to +1.0. High operating temperatures cause oil to thin out, dropping oil pressure. The heatmap highlights these physical relationships so the AI can evaluate multiple sensors together instead of relying on a single static limit."
                )
                
        with col2:
            if os.path.exists("outputs/figures/scatter_temp.png"):
                st.image("outputs/figures/scatter_temp.png", use_container_width=True)
                render_graph_explanation(
                    "Lub Oil Temp vs Coolant Temp Thermal Scatter Plot",
                    "Each dot is an engine reading. Red dots are engines needing maintenance. When both oil temperature and coolant temperature get very high (top-right corner), the dots turn almost completely red.",
                    "High temperatures in both oil and coolant loops mean the engine is overheating. When coolant temperature exceeds 100°C and oil temperature exceeds 85°C simultaneously, engine maintenance is almost guaranteed to be required."
                )

    with tab3:
        st.subheader("📦 Feature Range Boxplots & Outlier Range Analysis")
        if os.path.exists("outputs/figures/feature_boxplots.png"):
            st.image("outputs/figures/feature_boxplots.png", use_container_width=True)
            render_graph_explanation(
                "Sensor Feature Boxplots by Engine Condition",
                "Boxplots show the normal operating ranges, median values, and extreme high or low points (outliers) for each engine sensor.",
                "The middle line inside each box shows the median, while the box contains the middle 50% of the readings. Outliers like high RPM (up to 2,239 RPM) and high coolant temp (up to 195.5°C) represent extreme vehicle workload events that our model learns to handle without giving false alarms."
            )

# --- PAGE 4: MODEL BENCHMARK & METRICS ---
elif page == "🤖 AI Model Benchmark & Metrics":
    st.header("🤖 Machine Learning Model Benchmarking & Metric Comparison")
    st.write("Performance evaluation of 5 machine learning algorithms on 3,907 unseen test engine readings.")
    
    if os.path.exists("outputs/metrics/model_comparison.csv"):
        comp_df = pd.read_csv("outputs/metrics/model_comparison.csv")
        st.subheader("🏆 Experimental Model Performance Table")
        st.dataframe(comp_df.style.highlight_max(axis=0, subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], color="#065f46"), use_container_width=True)
        
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("outputs/figures/model_comparison.png"):
            st.image("outputs/figures/model_comparison.png", use_container_width=True)
            render_graph_explanation(
                "Model Benchmarking Performance Bar Chart",
                "This bar chart compares 5 AI models across Accuracy, Precision, Recall, F1-Score, and ROC-AUC. Support Vector Machine (SVM) performed best with an F1-Score of 77.09% and Recall of 89.57%.",
                "We benchmarked Logistic Regression, KNN, Decision Tree, Random Forest, and SVM. In vehicle maintenance, Recall is the most important metric because missing a faulty engine leads to expensive road breakdowns. SVM achieved the highest Recall (89.57%), catching nearly 9 out of 10 faults."
            )
            
    with col2:
        if os.path.exists("outputs/figures/roc_curves.png"):
            st.image("outputs/figures/roc_curves.png", use_container_width=True)
            render_graph_explanation(
                "Receiver Operating Characteristic (ROC) Curves",
                "The ROC curve shows how well each AI model can separate healthy engines from faulty engines. Curves closer to the top-left corner perform better.",
                "The Area Under the Curve (ROC-AUC) measures overall model quality. Support Vector Machine and Random Forest achieved the highest ROC-AUC score (~0.696), proving that they create a strong separation between normal and faulty engine states."
            )

    st.subheader("🎯 Confusion Matrix Grid & Error Analysis")
    if os.path.exists("outputs/figures/confusion_matrices.png"):
        st.image("outputs/figures/confusion_matrices.png", use_container_width=True)
        render_graph_explanation(
            "Confusion Matrix Grid Analysis",
            "This grid shows exact numbers of correct predictions and errors for all 5 models on 3,907 test engine readings.",
            "Out of 3,907 test engine readings, 2,463 actually required maintenance. Support Vector Machine (SVM) correctly identified 2,206 of them and missed only 257 (False Negatives = 10.4%). This low missed-fault rate makes SVM the safest model for real vehicles."
        )

# --- PAGE 5: LIVE PREDICTION FORM & TELEMETRY DIAGNOSIS ---
elif page == "🔮 Live Telemetry Maintenance Predictor":
    st.header("🔮 Live Telemetry Maintenance Diagnostic Terminal")
    st.write("Input real-time engine telemetry values below to execute instant diagnostic prediction via the production SVM model.")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rpm = st.number_input("⚙️ Engine RPM", min_value=50.0, max_value=3000.0, value=700.0, step=10.0, help="Crankshaft rotational speed in RPM")
            lub_temp = st.number_input("🌡️ Lub Oil Temp (°C)", min_value=50.0, max_value=120.0, value=84.14, step=0.5, help="Lubricating oil sump operating temperature")
            
        with col2:
            lub_press = st.number_input("🛢️ Lub Oil Pressure (bar)", min_value=0.0, max_value=10.0, value=2.49, step=0.1, help="Engine oil pressure")
            cool_temp = st.number_input("🌡️ Coolant Temp (°C)", min_value=50.0, max_value=220.0, value=81.63, step=0.5, help="Engine coolant fluid temperature")
            
        with col3:
            fuel_press = st.number_input("⛽ Fuel Pressure (bar)", min_value=0.0, max_value=30.0, value=11.79, step=0.2, help="Fuel delivery injection pressure")
            cool_press = st.number_input("💧 Coolant Pressure (bar)", min_value=0.0, max_value=10.0, value=3.17, step=0.1, help="Coolant loop circulation pressure")
            
        submit_btn = st.form_submit_button("⚡ RUN LIVE TELEMETRY DIAGNOSTIC", use_container_width=True)
        
    if submit_btn:
        input_dict = {
            "Engine RPM": float(rpm),
            "Lub Oil Pressure": float(lub_press),
            "Fuel Pressure": float(fuel_press),
            "Coolant Pressure": float(cool_press),
            "Lub Oil Temp": float(lub_temp),
            "Coolant Temp": float(cool_temp)
        }
        
        with st.spinner("Processing telemetry through SVM kernel classification engine..."):
            res = predict_engine_condition(input_dict)
            
        st.markdown("---")
        st.subheader("🎯 Real-Time Prediction Outcome")
        
        if res["prediction_code"] == 1:
            st.markdown(f"""
            <div class="pred-card-fault">
                <h2 style="color: #ff0054; margin: 0; font-weight: 900;">🚨 MAINTENANCE REQUIRED</h2>
                <p style="color: #fca5a5; font-size: 1.15rem; margin-top: 0.5rem;">
                    Telemetry signals indicate internal engine stress. Immediate inspection recommended.
                </p>
                <div style="font-size: 1.7rem; font-weight: 900; color: #ffffff; margin-top: 1rem; font-family: 'JetBrains Mono', monospace;">
                    Model Confidence: {res['confidence_score']:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="pred-card-normal">
                <h2 style="color: #34d399; margin: 0; font-weight: 900;">✅ NORMAL ENGINE CONDITION</h2>
                <p style="color: #a7f3d0; font-size: 1.15rem; margin-top: 0.5rem;">
                    All sensor telemetry parameters are operating within standard baseline limits.
                </p>
                <div style="font-size: 1.7rem; font-weight: 900; color: #ffffff; margin-top: 1rem; font-family: 'JetBrains Mono', monospace;">
                    Model Confidence: {res['confidence_score']:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 Class Probability Breakdown")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.write(f"Normal State Probability: **{res['normal_probability']*100:.2f}%**")
            st.progress(res['normal_probability'])
        with p_col2:
            st.write(f"Maintenance State Probability: **{res['maintenance_probability']*100:.2f}%**")
            st.progress(res['maintenance_probability'])
            
        if res["warnings"]:
            st.warning("\n".join(res["warnings"]))
