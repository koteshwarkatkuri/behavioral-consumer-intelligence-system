import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Behavioral Consumer Intelligence System",
    layout="wide"
)

# =====================================
# TITLE
# =====================================

st.title("🧠 Behavioral Consumer Intelligence System")

st.markdown("""
### AI-Powered Consumer Purchase Prediction Platform

This system uses:
- XGBoost Machine Learning
- Behavioral Analytics
- Explainable AI (SHAP)
- Real-Time Prediction Intelligence
""")

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("📌 System Information")

st.sidebar.info("""
Upload customer browsing dataset
to predict:

✅ Purchase Probability  
✅ Buyer Behavior Type  
✅ Consumer Intent  
✅ Decision Fatigue Patterns
""")

# =====================================
# LOAD MODEL
# =====================================

model = joblib.load(
    "behavioral_purchase_model.pkl"
)

model_features = joblib.load(
    "model_features.pkl"
)

# =====================================
# FILE UPLOAD
# =====================================

uploaded_file = st.file_uploader(
    "📂 Upload CSV Dataset",
    type=["csv"]
)

# =====================================
# MAIN PROCESSING
# =====================================

if uploaded_file is not None:

    try:

        # =====================================
        # READ DATASET
        # =====================================

        df = pd.read_csv(uploaded_file)

        # =====================================
        # FEATURE ENGINEERING
        # =====================================

        df['engagement_score'] = (
            df['Administrative_Duration'] +
            df['Informational_Duration'] +
            df['ProductRelated_Duration']
        )

        df['product_focus_ratio'] = (
            df['ProductRelated'] /
            (df['Administrative'] + 1)
        )

        df['quick_decision_score'] = (
            df['PageValues'] /
            (df['ProductRelated_Duration'] + 1)
        )

        df['browsing_efficiency'] = (
            df['PageValues'] /
            (df['BounceRates'] + 0.001)
        )

        df['exit_intent_score'] = (
            df['ExitRates'] *
            df['PageValues']
        )

        df['informational_neglect'] = (
            df['ProductRelated'] -
            df['Informational']
        )

        # =====================================
        # SHOW DATASET
        # =====================================

        st.subheader("📌 Uploaded Dataset Preview")

        st.dataframe(df.head())

        # =====================================
        # KEEP REQUIRED FEATURES
        # =====================================

        df = df[model_features]

        # =====================================
        # NUMERIC CONVERSION
        # =====================================

        for col in model_features:

            df[col] = pd.to_numeric(
                df[col],
                errors='coerce'
            )

        df = df.fillna(0)

        # =====================================
        # MODEL PREDICTION
        # =====================================

        predictions = model.predict(df)

        probabilities = model.predict_proba(df)[:, 1]

        # =====================================
        # RESULT DATAFRAME
        # =====================================

        result_df = df.copy()

        result_df['Purchase_Prediction'] = np.where(
            predictions == 1,
            'Likely Buyer',
            'Non-Buyer'
        )

        result_df['Purchase_Probability'] = (
            probabilities * 100
        ).round(2)

        # =====================================
        # BEHAVIOR LABELS
        # =====================================

        behavior_labels = []

        for prob in probabilities:

            if prob >= 0.70:

                behavior_labels.append(
                    "Focused Intent Buyer"
                )

            elif prob >= 0.40:

                behavior_labels.append(
                    "Moderate Intent Explorer"
                )

            elif prob >= 0.20:

                behavior_labels.append(
                    "Hesitant Researcher"
                )

            else:

                behavior_labels.append(
                    "Decision-Fatigued Explorer"
                )

        result_df["Behavior_Type"] = behavior_labels

        # =====================================
        # SHOW RESULTS
        # =====================================

        st.subheader("📊 Prediction Results")

        st.dataframe(result_df.head(20))

        # =====================================
        # METRICS
        # =====================================

        total_buyers = (
            result_df['Purchase_Prediction']
            == 'Likely Buyer'
        ).sum()

        total_non_buyers = (
            result_df['Purchase_Prediction']
            == 'Non-Buyer'
        ).sum()

        avg_probability = (
            result_df['Purchase_Probability']
            .mean()
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "🛒 Likely Buyers",
            total_buyers
        )

        col2.metric(
            "❌ Non Buyers",
            total_non_buyers
        )

        col3.metric(
            "📈 Avg Purchase Probability",
            f"{avg_probability:.2f}%"
        )

        # =====================================
        # CHART LAYOUT
        # =====================================

        chart_col1, chart_col2 = st.columns(2)

        # =====================================
        # BUYER DISTRIBUTION CHART
        # =====================================

        with chart_col1:

            st.subheader("📈 Buyer Distribution")

            buyer_counts = (
                result_df['Purchase_Prediction']
                .value_counts()
            )

            fig1, ax1 = plt.subplots(figsize=(4, 3))

            ax1.bar(
                buyer_counts.index,
                buyer_counts.values,
                color=['green', 'red']
            )

            ax1.set_title("Buyer Distribution")
            ax1.set_ylabel("Count")

            st.pyplot(fig1)

        # =====================================
        # BEHAVIOR DISTRIBUTION CHART
        # =====================================

        with chart_col2:

            st.subheader("🧠 Behavior Segmentation")

            behavior_counts = (
                result_df['Behavior_Type']
                .value_counts()
            )

            fig2, ax2 = plt.subplots(figsize=(5, 3))

            ax2.bar(
                behavior_counts.index,
                behavior_counts.values,
                color=[
                    'blue',
                    'orange',
                    'purple',
                    'brown'
                ]
            )

            ax2.set_title("Behavior Segments")
            ax2.set_ylabel("Count")

            plt.xticks(rotation=15, fontsize=8)

            st.pyplot(fig2)

        # =====================================
        # PURCHASE PROBABILITY HISTOGRAM
        # =====================================

        st.subheader("📈 Purchase Probability Distribution")

        fig3, ax3 = plt.subplots(figsize=(6, 3))

        ax3.hist(
            result_df['Purchase_Probability'],
            bins=20,
            color='skyblue',
            edgecolor='black'
        )

        ax3.set_xlabel("Purchase Probability")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Probability Histogram")

        st.pyplot(fig3)

        # =====================================
        # SHAP EXPLAINABILITY
        # =====================================

        st.subheader("🧠 Explainable AI - SHAP Feature Importance")

        explainer = shap.Explainer(model)

        shap_values = explainer(df)

        plt.figure(figsize=(7, 4))

        shap.plots.bar(
            shap_values,
            show=False
        )

        st.pyplot(plt.gcf())

        # =====================================
        # AI INSIGHTS
        # =====================================

        st.subheader("📌 AI Behavioral Insights")

        st.success("""
        ✅ High browsing efficiency increases purchase intent.

        ✅ Higher page values strongly influence buyer conversion.

        ✅ Customers with lower bounce rates show stronger buying behavior.

        ✅ Longer product-related duration indicates high engagement.

        ✅ Exit intent score helps identify purchase hesitation.
        """)

        # =====================================
        # DOWNLOAD RESULTS
        # =====================================

        csv = result_df.to_csv(index=False)

        st.download_button(
            label="⬇ Download Prediction Results",
            data=csv,
            file_name="behavioral_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(f"❌ Error: {e}")