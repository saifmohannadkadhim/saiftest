# -*- coding: utf-8 -*-
"""
AI Data Intelligence – ILO Social Security Campaign (Streamlit)
English UI • Auto-load English translation dataset • Auto column detection
Author: Insight Companion
"""

import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

# Optional Arabic display support for labels present in data values
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
except Exception:
    arabic_reshaper = None
    get_display = None

def reshape_ar(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    if arabic_reshaper and get_display:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            return text
    return text

# ============== CONFIG ==============
DEFAULT_FILE = "Annex 1-Activity Evaluation Electric Form.xlsx"  # English translation dataset

st.set_page_config(page_title="AI Data Intelligence – ILO", layout="wide")
st.title("🧠 AI Data Intelligence – Social Security Awareness Campaign")
st.caption("English UI • Auto-loaded dataset • Offline • Matplotlib only")

# ============== CONSTANTS ==============
AR_YES = {"نعم", "اي", "أي", "yes", "y", "true"}
AR_NO  = {"لا", "كلا", "no", "n", "false"}
AR_DK  = {"لا اعلم", "لا أعلم", "غير متأكد", "unknown", "i don't know", "idk", "dk"}

# Keywords for auto-detection (English + Arabic + your prior schema)
COLUMN_KEYWORDS = {
    "gender":       ["gender", "sex", "جنس"],
    "governorate":  ["governorate", "city", "govern", "محافظة", "المحافظة"],
    "employment":   ["employment", "status_job", "staus_job", "work", "job", "employ", "المهنة", "العمل", "حالة"],
    "heard_law":    ["heard", "hear", "سمعت", "law", "قانون", "الضمان"],
    "enrolled_ss":  ["registered", "register", "enroll", "مسجل", "تسجيل"],
    "intent_after": ["intention", "intend", "after_part", "رغبة", "بعد", "مشاركة"],
    "reason_text":  ["reason", "reseon", "why", "comment", "note", "سبب", "لماذا", "ملاحظة"],
}

STOPWORDS_AR = set("""
في من على الى إلى عن أن إن كان كانت الذين التي هذا هذه ذلك تلك ثم حيث كما اذا إذا مع لدى عند لهم لها به بها هو هي هم هن نحن أنت انتم لكن بل أو أم ولا ولم لما ما لا ليس دون ضد فوق تحت بين خلال بسبب حول قبل بعد جداً جدا فقط أكثر اقل مثل مثلما لأن لان كيف لماذا متى أين هنا هناك كل جميع أي شيء بعض كثير قليل ربما قد سيتم سوف حتى لدى لدي لديكم لدينا ألا إلا غير ضمن بسبباً بسبب
""".split())

# ============== HELPERS ==============
def normalize_text(val: str) -> str:
    if pd.isna(val): return ""
    s = str(val).strip().lower()
    s = s.replace("إ", "ا").replace("أ", "ا").replace("آ", "ا").replace("ة", "ه")
    s = re.sub(r"\s+", " ", s)
    return s

def auto_detect_columns(df: pd.DataFrame):
    mapping = {}
    for col in df.columns:
        norm = normalize_text(col)
        for key, keywords in COLUMN_KEYWORDS.items():
            if any(k in norm for k in keywords):
                mapping.setdefault(key, col)  # keep first match
    return mapping

def yes_no_dk(val: str) -> str:
    if pd.isna(val): return "Unknown"
    s = normalize_text(val)
    if s in AR_YES: return "Yes"
    if s in AR_NO: return "No"
    if s in AR_DK: return "Unknown"
    # light heuristics for English values
    if "yes" in s: return "Yes"
    if "no" in s:  return "No"
    if "unknown" in s: return "Unknown"
    return s or "Unknown"

def tokenize_mixed(text: str):
    # Keep Arabic words for Arabic NLP, and basic lowercased English tokens
    s = normalize_text(text)
    # Split by non-letter (Arabic or Latin)
    tokens = re.findall(r"[a-zA-Z]+|[\u0600-\u06FF]+", s)
    # Drop Arabic stopwords; keep English simple
    tokens = [t for t in tokens if not (re.match(r"^[\u0600-\u06FF]+$", t) and t in STOPWORDS_AR)]
    return tokens

def bar_plot(series: pd.Series, title: str, xlabel: str, reshape_labels=False):
    counts = series.value_counts(dropna=False)
    if reshape_labels:
        new_index = []
        for lab in counts.index.astype(str):
            if re.search(r"[\u0600-\u06FF]", lab):
                new_index.append(reshape_ar(lab))
            else:
                new_index.append(lab)
        counts.index = new_index
    fig = plt.figure()
    counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Participants")
    st.pyplot(fig)

# ============== DATA LOADING ==============
st.sidebar.subheader("📂 Data Source")
use_default = st.sidebar.checkbox("Use default English translation dataset", value=True)

df_raw = None
read_error = None

if use_default:
    try:
        df_raw = pd.read_excel(DEFAULT_FILE)
        st.success(f"Auto-loaded dataset: {os.path.basename(DEFAULT_FILE)}")
    except Exception as e:
        read_error = f"Failed to auto-load default file: {e}"

uploaded = st.sidebar.file_uploader("Or upload another Excel file", type=["xlsx", "xls"])
if uploaded is not None:
    try:
        df_raw = pd.read_excel(uploaded)
        st.success("Uploaded file loaded successfully.")
        read_error = None
        use_default = False
    except Exception as e:
        read_error = f"Failed to read uploaded file: {e}"

if df_raw is None:
    st.error(read_error or "No dataset available.")
    st.stop()

st.subheader("🔎 Raw Data Preview")
st.dataframe(df_raw.head(12))

# ============== COLUMN MAPPING ==============
detected = auto_detect_columns(df_raw)
st.success("✅ Auto-detected columns")
st.json(detected)

st.sidebar.markdown("---")
st.sidebar.subheader("🧭 Column Mapping (optional override)")
cols = list(df_raw.columns)

def selbox(label, key):
    default = detected.get(key)
    idx = cols.index(default) + 1 if default in cols else 0
    return st.sidebar.selectbox(label, options=[None] + cols, index=idx)

col_gender   = selbox("Gender", "gender")
col_govern   = selbox("Governorate / City", "governorate")
col_employ   = selbox("Employment Status", "employment")
col_heard    = selbox("Heard about the Law?", "heard_law")
col_enroll   = selbox("Registered in Social Security?", "enrolled_ss")
col_intent   = selbox("Intention to Register After Participation?", "intent_after")
col_reason   = selbox("Reason for NOT registering (optional)", "reason_text")

required = [col_gender, col_govern, col_employ, col_heard, col_enroll, col_intent]
if any(c is None for c in required):
    st.warning("Please ensure all required fields are mapped: gender, governorate, employment, heard_law, enrolled_ss, intent_after.")
    st.stop()

# ============== STANDARDIZED DATAFRAME ==============
df = pd.DataFrame()
df["gender"]       = df_raw[col_gender].astype(str)
df["governorate"]  = df_raw[col_govern].astype(str)
df["employment"]   = df_raw[col_employ].astype(str)
df["heard_law"]    = df_raw[col_heard].apply(yes_no_dk)
df["enrolled_ss"]  = df_raw[col_enroll].apply(yes_no_dk)
df["intent_after"] = df_raw[col_intent].apply(yes_no_dk)
df["reason_text"]  = df_raw[col_reason].astype(str).fillna("") if col_reason else ""

st.subheader("📋 Cleaned & Unified Data")
st.dataframe(df.head(20))

# ============== KPIs ==============
st.subheader("📈 Key Performance Indicators (KPIs)")
total = len(df)
aware_rate   = (df["heard_law"] == "Yes").mean() if total else 0
enroll_rate  = (df["enrolled_ss"] == "Yes").mean() if total else 0
intent_rate  = (df["intent_after"] == "Yes").mean() if total else 0
female_share = (df["gender"].str.contains("female|انثى|أنثى", case=False)).mean() if total else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Participants", f"{total:,}")
c2.metric("Aware of the Law", f"{aware_rate*100:.1f}%")
c3.metric("Currently Registered", f"{enroll_rate*100:.1f}%")
c4.metric("Intend to Register", f"{intent_rate*100:.1f}%")
c5.metric("Female Share", f"{female_share*100:.1f}%")

# ============== DESCRIPTIVE ANALYTICS ==============
st.subheader("📊 Descriptive Analytics")
col1, col2 = st.columns(2)
with col1:
    bar_plot(df["governorate"], "Participants by Governorate/City", "Governorate/City", reshape_labels=True)
with col2:
    bar_plot(df["gender"], "Participants by Gender", "Gender", reshape_labels=True)

col3, col4 = st.columns(2)
with col3:
    bar_plot(df["employment"], "Participants by Employment", "Employment Type", reshape_labels=True)
with col4:
    bar_plot(df["heard_law"], "Heard about the Law?", "Response")

col5, col6 = st.columns(2)
with col5:
    bar_plot(df["enrolled_ss"], "Registered in Social Security?", "Response")
with col6:
    bar_plot(df["intent_after"], "Intention to Register After Participation?", "Response")

# ============== SEGMENTATION (KMeans) ==============
st.subheader("🧩 Audience Segmentation (KMeans)")
seg_features = ["gender", "governorate", "employment", "heard_law", "enrolled_ss"]
ohe = OneHotEncoder(handle_unknown='ignore')
seg_ct = ColumnTransformer([("cat", ohe, seg_features)], remainder='drop')

k = st.slider("Number of clusters (k)", 2, 8, 4)
try:
    X_seg = seg_ct.fit_transform(df[seg_features])
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(X_seg)
    df["segment"] = labels

    st.write("Sample of segmented records:")
    st.dataframe(df[["gender","governorate","employment","heard_law","enrolled_ss","intent_after","segment"]].head(15))

    fig = plt.figure()
    df["segment"].value_counts().sort_index().plot(kind="bar")
    plt.title("Segment Sizes")
    plt.xlabel("Segment")
    plt.ylabel("Participants")
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Segmentation failed: {e}")

# ============== PREDICTIVE MODEL (RandomForest) ==============
st.subheader("🔮 Predictive Model – Intention to Register")
model_features = ["gender", "governorate", "employment", "heard_law", "enrolled_ss"]
target = "intent_after"

df_model = df[df[target].isin(["Yes", "No"])].copy()
if len(df_model) < 30:
    st.info("Not enough data to train a reliable model (need ≥ 30 rows with Yes/No target).")
else:
    X = df_model[model_features]
    y = df_model[target].apply(lambda v: 1 if v == "Yes" else 0)

    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown='ignore'), model_features)])
    clf = Pipeline([
        ("pre", pre),
        ("imp", SimpleImputer(strategy='most_frequent')),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=42))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds)

    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{acc*100:.1f}%")
    c2.metric("F1-Score", f"{f1*100:.1f}%")

    # Feature importances (approximate via RF importances)
    try:
        imp = clf.named_steps["rf"].feature_importances_
        ohe_fitted = clf.named_steps["pre"].named_transformers_["cat"]
        feat_names = ohe_fitted.get_feature_names_out(model_features)
        imp_df = pd.DataFrame({"Feature": feat_names, "Importance": imp}).sort_values("Importance", ascending=False).head(15)

        fig = plt.figure()
        imp_df.set_index("Feature")["Importance"].plot(kind="bar")
        plt.title("Top Predictive Features")
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not compute feature importances: {e}")

# ============== NLP (TF-IDF + optional LDA topics) ==============
st.subheader("🗣️ NLP – Reasons for NOT Registering")
reasons = df["reason_text"].dropna().astype(str)
if reasons.empty:
    st.info("No text responses available for NLP.")
else:
    try:
        tfidf = TfidfVectorizer(tokenizer=tokenize_mixed, ngram_range=(1,2), max_features=2000)
        X_tfidf = tfidf.fit_transform(reasons)
        means = np.asarray(X_tfidf.mean(axis=0)).ravel()
        vocab = np.array(tfidf.get_feature_names_out())
        top_idx = means.argsort()[::-1][:20]
        top_terms = pd.Series(means[top_idx], index=vocab[top_idx])

        # Reshape Arabic terms for display if any
        top_terms.index = [reshape_ar(t) if re.search(r"[\u0600-\u06FF]", t) else t for t in top_terms.index]

        fig = plt.figure()
        top_terms.sort_values(ascending=True).plot(kind="barh")
        plt.title("Top Keywords / Phrases")
        plt.xlabel("TF-IDF")
        plt.ylabel("Keyword")
        st.pyplot(fig)

        # Optional topic modeling
        n_topics = st.slider("Number of topics (LDA)", 2, 8, 3)
        count_vec = CountVectorizer(tokenizer=tokenize_mixed, max_features=3000)
        X_counts = count_vec.fit_transform(reasons)
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(X_counts)
        terms = np.array(count_vec.get_feature_names_out())
        st.markdown("**Topic summaries:**")
        for i, comp in enumerate(lda.components_):
            top = comp.argsort()[::-1][:10]
            topic_terms = [reshape_ar(t) if re.search(r"[\u0600-\u06FF]", t) else t for t in terms[top]]
            st.markdown(f"- Topic {i+1}: {', '.join(topic_terms)}")

    except Exception as e:
        st.warning(f"NLP failed: {e}")

# ============== DOWNLOAD ==============
st.subheader("⬇️ Download Cleaned Data")
out = df.to_csv(index=False).encode("utf-8-sig")
st.download_button("Download CSV", data=out, file_name="cleaned_participants.csv", mime="text/csv")




# ============== ADVANCED PREDICTIVE AI ==============
st.subheader("🤖 Advanced Predictive AI – Explainability & Simulation")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import shap

# Select candidate models for stacking / comparison
models = {
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

results = {}
best_model = None
best_f1 = 0
best_name = None

st.write("Training multiple models for comparison...")

for name, model in models.items():
    try:
        pipeline = Pipeline([
            ("pre", ColumnTransformer([("cat", OneHotEncoder(handle_unknown='ignore'), model_features)])),
            ("imp", SimpleImputer(strategy='most_frequent')),
            ("clf", model)
        ])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1s = f1_score(y_test, preds)
        results[name] = {"Accuracy": acc, "F1": f1s}
        if f1s > best_f1:
            best_model = pipeline
            best_name = name
            best_f1 = f1s
    except Exception as e:
        results[name] = {"Error": str(e)}

# Display model comparison
st.write("### 🧩 Model Comparison (AutoML Summary)")
st.dataframe(pd.DataFrame(results).T.style.format({"Accuracy": "{:.2%}", "F1": "{:.2%}"}))

if best_model:
    st.success(f"✅ Best model selected automatically: **{best_name}** (F1={best_f1:.2%})")

    # ============== Explainable AI (SHAP) ==============
    st.write("### 🧠 Explainable AI (Why the Model Predicts This)")
    try:
        # Extract fitted model and encoder
        ohe_fitted = best_model.named_steps["pre"].named_transformers_["cat"]
        feature_names = ohe_fitted.get_feature_names_out(model_features)
        model_estimator = best_model.named_steps["clf"]

        # Compute SHAP values
        explainer = shap.Explainer(model_estimator)
        shap_values = explainer(best_model.named_steps["pre"].transform(X_test[:200]))

        st.write("#### Feature Importance (SHAP Summary Plot)")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, features=pd.DataFrame(best_model.named_steps["pre"].transform(X_test[:200]),
                            columns=feature_names), show=False)
        
        st.pyplot(fig)
        
    except Exception as e:
        st.warning(f"SHAP explainability not available for this model: {e}")

    # ============== Prediction Simulator ==============
    st.write("### 🎛️ Prediction Simulator")
    st.caption("Change the values below to simulate a new participant and see predicted registration intention.")

    sim_gender = st.selectbox("Gender", df["gender"].unique(), index=0)
    sim_govern = st.selectbox("Governorate", df["governorate"].unique(), index=0)
    sim_employ = st.selectbox("Employment Type", df["employment"].unique(), index=0)
    sim_heard = st.selectbox("Heard about the Law?", ["Yes", "No", "Unknown"], index=0)
    sim_enroll = st.selectbox("Already Registered in SS?", ["Yes", "No", "Unknown"], index=1)

    sim_df = pd.DataFrame([{
        "gender": sim_gender,
        "governorate": sim_govern,
        "employment": sim_employ,
        "heard_law": sim_heard,
        "enrolled_ss": sim_enroll
    }])

    try:
        sim_pred_prob = best_model.predict_proba(sim_df)[0][1]
        st.metric("Predicted Probability of Willingness to Register", f"{sim_pred_prob*100:.1f}%")

        if sim_pred_prob >= 0.7:
            st.success("🟢 High likelihood: This profile is likely to register soon.")
        elif sim_pred_prob >= 0.4:
            st.warning("🟡 Moderate likelihood: Might register with further awareness.")
        else:
            st.error("🔴 Low likelihood: Low chance to register without intervention.")
    except Exception as e:
        st.warning(f"Prediction simulation failed: {e}")




