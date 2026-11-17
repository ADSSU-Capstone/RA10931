import io
import re
import string
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

st.set_page_config(
    page_title="Sentiment Analysis of Student Feedback on FHE and TES at ADSSU",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title { font-size: 2.0rem; font-weight: 800; color: #0b3d5c; margin-bottom: 0.1rem; }
    .main-sub { color: #4a6572; font-size: 0.93rem; margin-bottom: 1.0rem; }
    .kpi-card { background: #f4f8fb; border: 1px solid #d3e2ec; border-radius: 10px;
                padding: 14px 18px; margin-bottom: 8px; }
    .kpi-value { font-size: 2.05rem; font-weight: 800; color: #0b3d5c; line-height: 1.1; }
    .kpi-label { font-size: 0.8rem; color: #4a6572; text-transform: uppercase; letter-spacing: 0.6px; }
    .interp-box { background: #eef6f1; border-left: 5px solid #2e7d5b; border-radius: 6px;
                  padding: 10px 14px; font-size: 0.9rem; color: #1f3d2f; margin: 6px 0 18px 0; }
    .justify-box { background: #fdf6ec; border-left: 5px solid #b58122; border-radius: 6px;
                   padding: 10px 14px; font-size: 0.9rem; color: #4a3a12; margin-bottom: 16px; }
    .warn-box { background: #fdecec; border-left: 5px solid #b23b3b; border-radius: 6px;
                padding: 10px 14px; font-size: 0.9rem; color: #5c1f1f; margin-bottom: 16px; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; flex-wrap: wrap; }
    .stTabs [data-baseweb="tab"] { background: #e8eef3; border: 1px solid #c9d8e3;
        border-radius: 6px; padding: 8px 14px; font-weight: 600; color: #274b63; }
    .stTabs [aria-selected="true"] { background: #0b3d5c !important; color: #ffffff !important;
        border: 1px solid #0b3d5c !important; }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">Sentiment Analysis of Student Feedback on FHE and TES at Agusan del Sur State University</div>', unsafe_allow_html=True)
st.markdown('<div class="main-sub">A Data Mining Approach to Entry and Exit Survey Feedback on RA 10931 Programs</div>', unsafe_allow_html=True)

DEMO_COLS = {"B": "Age", "C": "Sex", "D": "Civil Status", "E": "Place of Residence",
             "G": "Type of Residence", "H": "College/Program", "I": "Year Level",
             "J": "Student Status", "K": "Awareness Channel"}

POSITIVE_LEXICON = {
    "helpful", "grateful", "thankful", "beneficial", "good", "excellent", "great",
    "happy", "satisfied", "appreciate", "appreciation", "positive", "love", "liked",
    "like", "easy", "accessible", "affordable", "save", "saved", "free", "support",
    "improved", "opportunity", "chance", "reduce", "reduced", "relief", "convenient",
    "better", "best", "perfect", "nice", "okay", "ok", "makakatulong", "salamat",
    "malaking tulong", "nakatulong", "mabuti", "maganda",
}
NEGATIVE_LEXICON = {
    "delay", "delayed", "problem", "issue", "difficult", "hard", "challenge",
    "not enough", "insufficient", "struggle", "struggling", "difficulty", "concern",
    "concerned", "poor", "lacking", "lack", "expensive", "costly", "confused",
    "unclear", "frustrat", "disappoint", "complaint", "slow", "late", "limited",
    "kulang", "mahal", "mahirap", "problema", "hindi sapat",
}

AGE_WHITELIST = ["18–27", "28–37", "38–47"]
RESIDENCE_WHITELIST = ["Agusan del Sur", "Agusan del Norte", "Surigao del Sur",
                       "Surigao del Norte", "Davao de Oro", "Other"]
LIKERT_MAP = {
    "1 - Strongly Disagree": 1, "1 - Never": 1, "2 - Disagree": 2, "2 - Rarely": 2,
    "3 - Neutral": 3, "3 - Sometimes": 3, "4 - Agree": 4, "4 - Frequently": 4,
    "4 - Often": 4, "5 - Strongly Agree": 5, "5 - Very Frequently": 5, "5 - Very Often": 5,
}


def strip_invisible(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    t = strip_invisible(text).lower()
    t = t.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in t.split() if w and w not in ENGLISH_STOP_WORDS and len(w) > 1]
    return " ".join(tokens)


def score_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return "Neutral", 0
    low = text.lower()
    pos = sum(1 for w in POSITIVE_LEXICON if w in low)
    neg = sum(1 for w in NEGATIVE_LEXICON if w in low)
    if pos > neg:
        return "Positive", pos - neg
    if neg > pos:
        return "Negative", neg - pos
    return "Neutral", 0


def map_likert(v):
    if not isinstance(v, str):
        return np.nan
    return LIKERT_MAP.get(strip_invisible(v), np.nan)


def to_num(s):
    return s.apply(map_likert)


@st.cache_data(show_spinner=False)
def read_upload(file_bytes, file_name):
    if file_name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=0)


def prep_survey(raw, label):
    df = raw.copy()
    report = {"survey": label, "raw": len(df)}
    df.columns = [strip_invisible(str(c)) for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].apply(strip_invisible)
            
    # Rename long question headers back to standard letter columns
    col_map = {}
    for c in df.columns:
        c_low = c.lower()
        if "age" in c_low and "B" not in col_map.values(): col_map[c] = "B"
        elif "sex" in c_low and "C" not in col_map.values(): col_map[c] = "C"
        elif "civil status" in c_low and "D" not in col_map.values(): col_map[c] = "D"
        elif "place of residence" in c_low and "E" not in col_map.values(): col_map[c] = "E"
        elif "type of residence" in c_low and "G" not in col_map.values(): col_map[c] = "G"
        elif "college" in c_low and "H" not in col_map.values(): col_map[c] = "H"
        elif "year level" in c_low and "I" not in col_map.values(): col_map[c] = "I"
        elif "student status" in c_low and "J" not in col_map.values(): col_map[c] = "J"
        elif "learn about" in c_low and "K" not in col_map.values(): col_map[c] = "K"
        
        # Map targets and satisfaction
        elif "overall" in c_low and "rate" in c_low and "AY" not in col_map.values(): col_map[c] = "AY"
        elif "satisfied" in c_low and "implementation" in c_low and "AZ" not in col_map.values(): col_map[c] = "AZ"
        
    df = df.rename(columns=col_map)

    report["duplicates"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    
    # Detect text columns dynamically
    text_keywords = ["experience", "like most", "difficulties", "concern", "improvement", "suggestion"]
    text_cols = [c for c in df.columns if any(k in c.lower() for k in text_keywords)]
    
    likert = [c for c in df.columns if c in list("LMNOPQRSTUV") + [
        "V","W","X","Y","Z","AA","AB","AC","AD","AE","AF","AG","AH","AI","AJ","AK","AL",
        "AM","AN","AO","AP","AQ","AR","AS","AT","AU","AV","AW","AX"]]
    if likert:
        num = df[likert].apply(to_num)
        drop = num.isna().all(axis=1)
        report["dropped_all_likert"] = int(drop.sum())
        df = df.loc[~drop].reset_index(drop=True)
        num = num.loc[~drop].reset_index(drop=True)
        aw = [c for c in list("LMNOPQRSTUV") if c in num.columns]
        fi = [c for c in ["V","W","X","Y","Z","AA","AB","AC","AD"] if c in num.columns]
        qu = [c for c in ["AE","AF","AG","AH","AI","AJ","AK","AL","AM","AN"] if c in num.columns]
        ch = [c for c in ["AO","AP","AQ","AR","AS","AT","AU","AV","AW","AX"] if c in num.columns]
        df["Awareness_Score"] = num[aw].mean(axis=1) if aw else np.nan
        df["Financial_Score"] = num[fi].mean(axis=1) if fi else np.nan
        df["Quality_Score"] = num[qu].mean(axis=1) if qu else np.nan
        df["Challenge_Score"] = (6 - num[ch]).mean(axis=1) if ch else np.nan
    else:
        report["dropped_all_likert"] = 0
        for c in ["Awareness_Score","Financial_Score","Quality_Score","Challenge_Score"]:
            df[c] = np.nan
            
    if "E" in df.columns:
        df["E"] = df["E"].where(df["E"].isin(RESIDENCE_WHITELIST), "Other")
    if "B" in df.columns:
        df["B"] = df["B"].where(df["B"].isin(AGE_WHITELIST), np.nan)
        
    df["Overall_Rating"] = to_num(df["AY"]) if "AY" in df.columns else np.nan
    df["Satisfaction"] = to_num(df["AZ"]) if "AZ" in df.columns else np.nan
    
    if text_cols:
        combined_text = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    else:
        combined_text = pd.Series([""] * len(df))
        
    df["Feedback_Text"] = combined_text
    df["Feedback_Clean"] = combined_text.apply(normalize_text)
    df["Sentiment"] = combined_text.apply(lambda x: score_sentiment(x)[0])
    df["Sentiment_Score"] = combined_text.apply(lambda x: score_sentiment(x)[1])
    df["Survey"] = label
    report["final"] = len(df)
    report["retention"] = report["final"] / max(report["raw"], 1)
    return df, report


def interp(text):
    st.markdown(f'<div class="interp-box">{text}</div>', unsafe_allow_html=True)


def justify(text):
    st.markdown(f'<div class="justify-box">{text}</div>', unsafe_allow_html=True)


def warn(text):
    st.markdown(f'<div class="warn-box">{text}</div>', unsafe_allow_html=True)


def kpi(value, label):
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{value}</div><div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)


st.sidebar.markdown("### Survey Uploads")
entry_up = st.sidebar.file_uploader("Entry Survey (CSV/Excel)", type=["csv","xlsx","xls"], key="entry")
exit_up = st.sidebar.file_uploader("Exit Survey (CSV/Excel)", type=["csv","xlsx","xls"], key="exit")

if entry_up is None or exit_up is None:
    st.info("Both the Entry Survey and the Exit Survey must be uploaded before the dashboard can load. No sample or example dataset is provided.")
    st.stop()

try:
    entry_raw = read_upload(entry_up.getvalue(), entry_up.name)
    exit_raw = read_upload(exit_up.getvalue(), exit_up.name)
except Exception as e:
    st.error(f"Unable to read one of the uploaded files: {e}")
    st.stop()

entry_df, entry_rep = prep_survey(entry_raw, "Entry")
exit_df, exit_rep = prep_survey(exit_raw, "Exit")

if len(entry_df) == 0 or len(exit_df) == 0:
    st.info("One or both surveys contain no valid records after preprocessing.")
    st.stop()

combined = pd.concat([entry_df, exit_df], ignore_index=True, sort=False)
res_opts = ["All"] + sorted(combined["E"].dropna().unique().tolist()) if "E" in combined.columns else ["All"]
focus = st.sidebar.selectbox("Place of Residence (drill-down)", res_opts)
entry_f = entry_df if focus == "All" else entry_df[entry_df["E"] == focus]
exit_f = exit_df if focus == "All" else exit_df[exit_df["E"] == focus]
filtered = combined if focus == "All" else combined[combined["E"] == focus]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Entry:** {entry_rep['final']} valid / {entry_rep['raw']} raw")
st.sidebar.markdown(f"**Exit:** {exit_rep['final']} valid / {exit_rep['raw']} raw")

tabs = st.tabs([
    "Data Preprocessing & Validation",
    "Demographic & Educational Profile",
    "Perceptions & Experiences (Entry vs Exit)",
    "Sentiment Distribution",
    "Sentiment Classification Model Comparison",
    "Recurring Themes & Issues",
    "Determinants of Satisfaction",
    "Patterns, Correlations & Clustering",
    "Decision Support & Recommendations",
])

with tabs[0]:
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: kpi(entry_rep["raw"], "Entry Raw")
    with c2: kpi(exit_rep["raw"], "Exit Raw")
    with c3: kpi(entry_rep["duplicates"]+exit_rep["duplicates"], "Duplicates Removed")
    with c4: kpi(entry_rep["dropped_all_likert"]+exit_rep["dropped_all_likert"], "All-Likert-Missing Dropped")
    with c5: kpi(entry_rep["final"], "Entry Valid")
    with c6: kpi(exit_rep["final"], "Exit Valid")
    justify(
        "Cleaning executed independently per survey: duplicate-row removal; stripping of <br> tags and invisible Unicode; "
        "whitespace collapse; removal of records missing all Likert responses; age restricted to 18–27, 28–37, 38–47; "
        "residence mapped to Agusan del Sur, Agusan del Norte, Surigao del Sur, Surigao del Norte, Davao de Oro, Other; "
        "N/A and blanks treated as missing. Open-ended text columns are detected dynamically and concatenated into a single "
        "feedback corpus, normalized, and labeled Positive / Neutral / Negative using the paper's specified lexicons."
    )
    rules = pd.DataFrame({
        "Rule": ["Duplicate removal","Whitespace/tag cleaning","All-Likert-missing drop",
                 "Age whitelist","Residence whitelist","N/A/blank → missing","Text corpus build",
                 "Sentiment labeling"],
        "Applied To": ["Both surveys","Both surveys, text","Likert blocks",
                       "Column B","Column E","Rating/Likert columns","Open-ended questions",
                       "Feedback_Clean"],
        "Entry Effect": [f"{entry_rep['duplicates']} removed","Stripped",f"{entry_rep['dropped_all_likert']} removed",
                         "Out-of-range→NaN","Non-whitelist→Other","Mapped to NaN","Concatenated","Lexicon-labeled"],
        "Exit Effect": [f"{exit_rep['duplicates']} removed","Stripped",f"{exit_rep['dropped_all_likert']} removed",
                        "Out-of-range→NaN","Non-whitelist→Other","Mapped to NaN","Concatenated","Lexicon-labeled"],
    })
    st.dataframe(rules, use_container_width=True, hide_index=True)
    interp(f"Entry retention {entry_rep['retention']*100:.1f}% and Exit retention {exit_rep['retention']*100:.1f}%. "
           f"Exit text corpus available for {int((exit_df['Feedback_Text'].str.len()>0).sum())} respondents, which is the "
           "primary dataset for sentiment classification and modeling.")

with tabs[1]:
    if "E" in combined.columns:
        res = combined.groupby(["E","Survey"]).size().reset_index(name="Respondents")
        bar = px.bar(res, x="Respondents", y="E", color="Survey", orientation="h", barmode="group",
                     title="Respondent Count by Place of Residence and Survey Wave")
        bar.update_layout(height=400)
        st.plotly_chart(bar, use_container_width=True)
        interp("Agusan del Sur anchors both waves because it hosts the university; sparse units produce unstable per-area comparisons.")
    demo_cols = [c for c in DEMO_COLS if c in filtered.columns]
    if demo_cols:
        chosen = st.selectbox("Demographic variable", demo_cols, format_func=lambda x: DEMO_COLS[x])
        vc = filtered.groupby([chosen,"Survey"]).size().reset_index(name="Count")
        bar2 = px.bar(vc, x=chosen, y="Count", color="Survey", barmode="group",
                      title=f"Distribution of {DEMO_COLS[chosen]} by Wave")
        bar2.update_layout(height=400)
        st.plotly_chart(bar2, use_container_width=True)
        interp(f"Within the current selection ({focus}), subgroup composition is shown side-by-side for Entry and Exit. "
               "Disparate subgroup sizes across waves caution against treating wave-to-wave differences as purely programmatic effects.")
    if {"E","Survey","Overall_Rating"}.issubset(filtered.columns):
        matrix = filtered.pivot_table(index="E", columns=["Survey","I"] if "I" in filtered.columns else ["Survey"],
                                      values="Overall_Rating", aggfunc="count", fill_value=0)
        st.markdown("**Respondent Count per Residence by Wave and Year Level**")
        st.dataframe(matrix, use_container_width=True)
        interp("Sparse cells signal that year-level conclusions within that residence unit are weakly supported.")

with tabs[2]:
    comp = [c for c in ["Awareness_Score","Financial_Score","Quality_Score","Challenge_Score",
                         "Overall_Rating","Satisfaction"] if c in combined.columns]
    if comp:
        means = combined.groupby("Survey")[comp].mean().T.reset_index()
        means.columns = ["Metric","Entry","Exit"]
        means["Delta"] = means["Exit"] - means["Entry"]
        st.dataframe(means.round(3), use_container_width=True, hide_index=True)
        melt = means.melt(id_vars="Metric", var_name="Survey", value_name="Mean")
        bar = px.bar(melt, x="Metric", y="Mean", color="Survey", barmode="group",
                     title="Mean Composite and Target Metrics by Wave")
        bar.update_layout(height=440, xaxis_tickangle=-35)
        st.plotly_chart(bar, use_container_width=True)
        biggest = means.reindex(means["Delta"].abs().sort_values(ascending=False).index).iloc[0]
        interp(f"Largest wave-to-wave shift: {biggest['Metric']} (Δ = {biggest['Delta']:+.3f}). Positive deltas indicate "
               "improvement over the implementation window; negative deltas flag regressed constructs.")
    if {"E","Survey","Overall_Rating"}.issubset(combined.columns):
        area = combined.groupby(["E","Survey"])["Overall_Rating"].mean().reset_index()
        piv = area.pivot(index="E", columns="Survey", values="Overall_Rating")
        if {"Entry","Exit"}.issubset(piv.columns):
            piv["Delta"] = piv["Exit"] - piv["Entry"]
            piv = piv.sort_values("Delta").reset_index()
            fig = px.bar(piv, x="Delta", y="E", orientation="h",
                         title="Change in Mean Overall Rating by Residence (Exit − Entry)",
                         color="Delta", color_continuous_scale="RdYlGn")
            fig.update_layout(height=380, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            interp(f"{piv.iloc[-1]['E']} records the strongest gain (Δ = {piv.iloc[-1]['Delta']:+.3f}); "
                   f"{piv.iloc[0]['E']} the weakest (Δ = {piv.iloc[0]['Delta']:+.3f}). Negative-end units are priority for follow-up.")
    justify("Wave comparison uses arithmetic means of composites and targets on the shared 1–5 scale. Because waves are not guaranteed "
            "to be the same individuals, deltas describe a population-level shift rather than a within-person change.")

with tabs[3]:
    sent_counts = exit_f["Sentiment"].value_counts().reindex(["Positive","Neutral","Negative"]).fillna(0)
    total = max(int(sent_counts.sum()), 1)
    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi(total, "Exit Responses Analyzed")
    with c2: kpi(f"{sent_counts['Positive']/total*100:.1f}%", "Positive")
    with c3: kpi(f"{sent_counts['Neutral']/total*100:.1f}%", "Neutral")
    with c4: kpi(f"{sent_counts['Negative']/total*100:.1f}%", "Negative")
    pie = px.pie(values=sent_counts.values, names=sent_counts.index,
                 title="Sentiment Distribution of Exit Survey Feedback",
                 color=sent_counts.index,
                 color_discrete_map={"Positive":"#2e7d5b","Neutral":"#b58122","Negative":"#b23b3b"})
    pie.update_layout(height=400)
    st.plotly_chart(pie, use_container_width=True)
    dom = sent_counts.idxmax()
    interp(f"The dominant sentiment class is {dom} ({sent_counts[dom]/total*100:.1f}%). "
           "Positive dominance indicates favorable overall perception; the presence of a non-trivial Negative share "
           "identifies respondents whose concerns require targeted review.")
    if "E" in exit_f.columns:
        by_area = exit_f.groupby(["E","Sentiment"]).size().reset_index(name="Count")
        bar = px.bar(by_area, x="E", y="Count", color="Sentiment", barmode="stack",
                     title="Sentiment Distribution by Place of Residence",
                     color_discrete_map={"Positive":"#2e7d5b","Neutral":"#b58122","Negative":"#b23b3b"})
        bar.update_layout(height=400)
        st.plotly_chart(bar, use_container_width=True)
        interp("Stacked bars reveal residence units whose Negative share is disproportionate. "
               "Those units are the priority targets for program communication and service improvement.")
    st.markdown("**Sample Sentiment-Labeled Responses**")
    sample = exit_f[exit_f["Feedback_Text"].str.len()>0][["E","Sentiment","Feedback_Text"]].head(10)
    st.dataframe(sample, use_container_width=True, hide_index=True)
    justify("Sentiment labeling applies the paper's specified lexicons. Positive cues: helpful, grateful, thankful, beneficial, "
            "good, excellent. Negative cues: delay, problem, issue, difficult, hard, challenge, not enough. Neutral when no cue "
            "dominates. Responses with equal positive and negative cue counts are labeled Neutral.")

with tabs[4]:
    data = exit_df[exit_df["Feedback_Clean"].str.len()>0][["Feedback_Clean","Sentiment"]].copy()
    if len(data) < 15 or data["Sentiment"].nunique() < 2:
        warn("Insufficient labeled Exit Survey text to train and compare the five classifiers. "
             "At least 15 responses and at least two sentiment classes are required.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            data["Feedback_Clean"], data["Sentiment"], test_size=0.3, random_state=42,
            stratify=data["Sentiment"] if data["Sentiment"].value_counts().min() >= 2 else None)
        vec = TfidfVectorizer(max_features=1500, ngram_range=(1,2))
        Xtr = vec.fit_transform(X_train)
        Xte = vec.transform(X_test)
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Multinomial Naive Bayes": MultinomialNB(),
            "Linear SVM": CalibratedClassifierCV(LinearSVC(max_iter=5000), cv=3),
            "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        }
        rows = []
        for name, mdl in models.items():
            mdl.fit(Xtr, y_train)
            pred = mdl.predict(Xte)
            rows.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision (macro)": precision_score(y_test, pred, average="macro", zero_division=0),
                "Recall (macro)": recall_score(y_test, pred, average="macro", zero_division=0),
                "F1 (macro)": f1_score(y_test, pred, average="macro", zero_division=0),
            })
        results = pd.DataFrame(rows).sort_values("F1 (macro)", ascending=False).reset_index(drop=True)
        st.dataframe(results.round(3), use_container_width=True, hide_index=True)
        bar = px.bar(results.melt(id_vars="Model", var_name="Metric", value_name="Score"),
                     x="Model", y="Score", color="Metric", barmode="group",
                     title="Five-Algorithm Sentiment Classification Comparison")
        bar.update_layout(height=420)
        st.plotly_chart(bar, use_container_width=True)
        best = results.iloc[0]
        interp(f"Best-performing classifier on macro-F1: {best['Model']} "
               f"(Accuracy {best['Accuracy']:.3f}, Precision {best['Precision (macro)']:.3f}, "
               f"Recall {best['Recall (macro)']:.3f}, F1 {best['F1 (macro)']:.3f}). "
               "Macro averaging is used because class sizes are not guaranteed to be equal and each "
               "sentiment class must be credited equally.")
        best_model = models[best["Model"]]
        best_pred = best_model.predict(Xte)
        labels = sorted(data["Sentiment"].unique())
        cm = confusion_matrix(y_test, best_pred, labels=labels)
        cm_fig = px.imshow(cm, text_auto=True, aspect="auto", x=labels, y=labels,
                           labels={"x":"Predicted","y":"Actual"},
                           color_continuous_scale="Blues",
                           title=f"Confusion Matrix — {best['Model']}")
        cm_fig.update_layout(height=400)
        st.plotly_chart(cm_fig, use_container_width=True)
        dist = data["Sentiment"].value_counts()
        if dist.min()/dist.max() < 0.67:
            warn(f"Class distribution is imbalanced ({dict(dist)}). Macro-averaged precision, recall, and F1 are "
                 "the appropriate comparison metrics; raw accuracy is inflated by the majority class.")
        else:
            interp(f"Class distribution is approximately balanced ({dict(dist)}). Accuracy is a reasonable headline "
                   "metric, with macro precision and recall providing complementary views.")
        justify("Five classifiers are compared per Chapter 1 Objective 3: Logistic Regression, Multinomial Naive Bayes, "
                "Linear SVM (calibrated for probability outputs), Random Forest, and Gradient Boosting. Features are "
                "TF-IDF unigrams and bigrams (1,500 max features). A stratified 70/30 split preserves class ratios. "
                "Evaluation uses accuracy, macro precision, macro recall, macro F1, and a 3×3 confusion matrix.")

with tabs[5]:
    texts = exit_df[exit_df["Feedback_Clean"].str.len()>0]["Feedback_Clean"].tolist()
    if not texts:
        warn("No normalized Exit Survey text available for theme extraction.")
    else:
        all_tokens = []
        for t in texts:
            all_tokens.extend(t.split())
        freq = Counter(all_tokens).most_common(25)
        freq_df = pd.DataFrame(freq, columns=["Term","Frequency"])
        bar = px.bar(freq_df.sort_values("Frequency"), x="Frequency", y="Term", orientation="h",
                     title="Top 25 Terms in Exit Survey Feedback")
        bar.update_layout(height=600)
        st.plotly_chart(bar, use_container_width=True)
        interp("High-frequency terms identify the vocabulary dominating student feedback. "
               "Themes such as tuition, allowance, transportation, and facilities typically surface here.")
        theme_lex = {
            "Allowance / stipend delays": ["stipend","allowance","delay","delayed","late"],
            "Transportation & daily expenses": ["transportation","fare","food","daily","expenses"],
            "School supplies & projects": ["supplies","project","projects","printing","materials"],
            "Information dissemination": ["information","orientation","announcement","updates","clear"],
            "Facilities & learning resources": ["facilities","resources","internet","library","laboratory"],
            "Staff responsiveness": ["office","personnel","staff","response","assistance","service"],
        }
        theme_counts = {}
        for theme, words in theme_lex.items():
            theme_counts[theme] = sum(1 for t in texts if any(w in t for w in words))
        theme_df = pd.DataFrame(list(theme_counts.items()), columns=["Theme","Responses"]).sort_values("Responses")
        bar2 = px.bar(theme_df, x="Responses", y="Theme", orientation="h",
                      title="Recurring Themes Across Exit Survey Feedback",
                      color="Responses", color_continuous_scale="Teal")
        bar2.update_layout(height=420, coloraxis_showscale=False)
        st.plotly_chart(bar2, use_container_width=True)
        top_theme = theme_df.iloc[-1]
        interp(f"The most frequently recurring theme is '{top_theme['Theme']}' appearing in "
               f"{top_theme['Responses']} responses. This theme is the priority operational lever for program improvement.")
        justify("Theme extraction uses a domain lexicon drawn from the recurring issues identified in Chapter 1 "
                "(allowance, transportation, supplies, information, facilities, staff). Each theme is counted once per "
                "response if any of its cue terms appear, producing a mutually non-exclusive frequency profile.")

with tabs[6]:
    det_cols = [c for c in ["Awareness_Score","Financial_Score","Quality_Score","Challenge_Score","Overall_Rating"] if c in exit_df.columns]
    if "Satisfaction" in exit_df.columns and len(det_cols) >= 2:
        data = exit_df[det_cols + ["Satisfaction"]].dropna()
        if len(data) >= 15:
            corr = data.corr(method="pearson")["Satisfaction"].drop("Satisfaction").sort_values()
            corr_df = corr.reset_index()
            corr_df.columns = ["Determinant","Pearson r with Satisfaction"]
            bar = px.bar(corr_df, x="Pearson r with Satisfaction", y="Determinant", orientation="h",
                         title="Correlation of Each Determinant with Satisfaction",
                         color="Pearson r with Satisfaction", color_continuous_scale="RdBu_r",
                         range_color=[-1,1])
            bar.update_layout(height=380, coloraxis_showscale=False)
            st.plotly_chart(bar, use_container_width=True)
            top_pos = corr.idxmax()
            top_neg = corr.idxmin()
            interp(f"Strongest positive determinant of satisfaction: {top_pos} (r = {corr.max():.2f}). "
                   f"Strongest negative determinant: {top_neg} (r = {corr.min():.2f}). "
                   "Determinants with |r| below 0.2 are weak and should not be treated as primary levers.")
        else:
            warn("Insufficient valid records to compute satisfaction determinants.")
    else:
        warn("Satisfaction column or composite predictors are unavailable.")
    justify("Determinants are quantified via Pearson correlation between each composite predictor and the satisfaction "
            "field, directly answering Chapter 1 Objective 4. Pearson is appropriate because satisfaction and the "
            "composites are continuous scale-consistent measures and the objective is a linear association description.")

with tabs[7]:
    corr_cols = [c for c in ["Awareness_Score","Financial_Score","Quality_Score","Challenge_Score","Overall_Rating"] if c in filtered.columns]
    if len(corr_cols) >= 2:
        corr = filtered[corr_cols].corr(method="pearson")
        heat = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         title="Pearson Correlation Matrix of Composite Scores")
        heat.update_layout(height=440)
        st.plotly_chart(heat, use_container_width=True)
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
        if not upper.empty:
            interp(f"Strongest positive pair: {upper.idxmax()[0]} vs {upper.idxmax()[1]} (r = {upper.max():.2f}). "
                   f"Strongest negative pair: {upper.idxmin()[0]} vs {upper.idxmin()[1]} (r = {upper.min():.2f}).")
    cluster_cols = [c for c in ["Awareness_Score","Financial_Score","Quality_Score","Challenge_Score"] if c in filtered.columns]
    if len(cluster_cols) >= 2 and len(filtered) >= 3:
        X = filtered[cluster_cols].dropna()
        if len(X) >= 3:
            k = st.slider("Number of clusters (K-Means)", 2, 6, 3)
            Xs = StandardScaler().fit_transform(X)
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xs)
            cdf = X.copy()
            cdf["Cluster"] = labels
            fig = px.scatter(cdf, x=cluster_cols[0], y=cluster_cols[1], color=cdf["Cluster"].astype(str),
                             title=f"K-Means Clustering (K={k})", opacity=0.75)
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(cdf.groupby("Cluster")[cluster_cols].mean().round(2), use_container_width=True)
            interp(f"K-Means on standardized composites with K={k} partitions students into segments whose centroids are "
                   "shown. Segments combining high quality and low reported challenges represent the most positive student profile.")
        justify("K-Means is applied per Chapter 1 Objective 5 to group respondents with similar characteristics, experiences, "
                "and perceptions. Standardization is required because composite scales, though nominally identical, "
                "carry different variance profiles.")

with tabs[8]:
    dec_cols = [c for c in ["Awareness_Score","Financial_Score","Quality_Score","Challenge_Score"] if c in filtered.columns]
    if len(dec_cols) >= 2 and "Overall_Rating" in filtered.columns and "E" in filtered.columns:
        data = filtered[dec_cols + ["Overall_Rating","E"]].dropna()
        if len(data) >= 15 and data["Overall_Rating"].nunique() >= 2:
            median = data["Overall_Rating"].median()
            y = (data["Overall_Rating"] > median).astype(int)
            X = data[dec_cols]
            if y.nunique() >= 2:
                mdl = RandomForestClassifier(n_estimators=300, random_state=42).fit(X, y)
                data = data.copy()
                data["Predicted_High_Probability"] = mdl.predict_proba(X)[:,1]
                area = data.groupby("E")["Predicted_High_Probability"].mean().reset_index()
                area.columns = ["Place of Residence","Mean Predicted Probability"]
                area = area.sort_values("Mean Predicted Probability")
                bar = px.bar(area, x="Mean Predicted Probability", y="Place of Residence", orientation="h",
                             title="Mean Predicted Probability of High Overall Rating by Residence",
                             color="Mean Predicted Probability", color_continuous_scale="Teal")
                bar.update_layout(height=380, coloraxis_showscale=False)
                st.plotly_chart(bar, use_container_width=True)
                top = area.iloc[-1]
                interp(f"{top['Place of Residence']} shows the highest mean predicted probability "
                       f"({top['Mean Predicted Probability']:.3f}). Areas at the bottom are priority for intervention.")
                out = data[["E"] + dec_cols + ["Predicted_High_Probability"]]
                out.columns = ["Place of Residence"] + dec_cols + ["Predicted_High_Probability"]
                st.download_button("Download Record-Level Predictions (CSV)",
                                   data=out.to_csv(index=False).encode("utf-8"),
                                   file_name="ra10931_predictions.csv", mime="text/csv")
                st.dataframe(out.head(20), use_container_width=True, hide_index=True)
        else:
            warn("Insufficient valid records for decision support.")
    st.markdown("**Program Recommendations Derived from the Results**")
    rec_rows = []
    if "Sentiment" in exit_df.columns:
        dist = exit_df["Sentiment"].value_counts(normalize=True)
        if dist.get("Negative", 0) > 0.25:
            rec_rows.append({"Area":"Program experience",
                             "Finding":f"{dist['Negative']*100:.1f}% negative sentiment share",
                             "Recommendation":"Targeted review of negative feedback to isolate recurring operational failures."})
    if "Challenge_Score" in exit_df.columns and exit_df["Challenge_Score"].mean() < 3:
        rec_rows.append({"Area":"Implementation challenges",
                         "Finding":f"Mean reverse-scored challenges {exit_df['Challenge_Score'].mean():.2f} below midpoint",
                         "Recommendation":"Strengthen frontline support and shorten response times to reported concerns."})
    if "Satisfaction" in exit_df.columns:
        rec_rows.append({"Area":"Satisfaction",
                         "Finding":f"Mean satisfaction {exit_df['Satisfaction'].mean():.2f}",
                         "Recommendation":"Sustain high-rated services; focus improvement on low-rated service touchpoints."})
    if rec_rows:
        st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)
    justify("Recommendations are surfaced only when a computed result crosses a defensible threshold: negative sentiment "
            "share above 25%, or mean reverse-scored challenges below the 3-point midpoint. This prevents speculative or "
            "non-data-driven advice from entering the defense.")
