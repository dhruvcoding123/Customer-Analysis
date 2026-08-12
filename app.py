import streamlit as st
import pandas as pd
import plotly.express as px

from data_loader import load_data
from signal_engine import (
    build_customer_signals,
    get_customer_transcripts,
)
from risk_scorer import add_structured_risk_score
from llm_analyzer import analyze_customer


st.set_page_config(
    page_title="Intelligent Customer Signal Detector",
    page_icon="🎯",
    layout="wide",
)

st.title(
    "🎯 Intelligent Customer Signal Detector"
)

st.caption(
    "AI-powered early warning system for customer "
    "retention and operations teams"
)

# Data preparation

@st.cache_data
def load_and_prepare_data():

    data = load_data()

    customer_signals = build_customer_signals(
        data
    )
    customer_signals = add_structured_risk_score(
        customer_signals
    )

    return data, customer_signals

data, customers = load_and_prepare_data()

# Risk overview

st.subheader("Customer Risk Overview")

col1, col2, col3, col4 = st.columns(4)
total_customers = len(customers)

high_risk = len(
    customers[
        customers["risk_level"] == "High"
    ]
)

medium_risk = len(
    customers[
        customers["risk_level"] == "Medium"
    ]
)

low_risk = len(
    customers[
        customers["risk_level"] == "Low"
    ]
)

col1.metric(
    "Customers Monitored",
    f"{total_customers:,}"
)

col2.metric(
    "🔴 High Risk",
    f"{high_risk:,}"
)

col3.metric(
    "🟠 Medium Risk",
    f"{medium_risk:,}"
)

col4.metric(
    "🟢 Low Risk",
    f"{low_risk:,}"
)


st.divider()

# Charts

left, right = st.columns(2)

with left:

    risk_data = pd.DataFrame({
        "Risk Level": [
            "High",
            "Medium",
            "Low",
        ],
        "Customers": [
            high_risk,
            medium_risk,
            low_risk,
        ],
    })

    fig = px.bar(
        risk_data,
        x="Risk Level",
        y="Customers",
        title="Customer Risk Distribution",
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


with right:

    fig = px.histogram(
        customers,
        x="structured_risk_score",
        nbins=20,
        title="Structured Risk Score Distribution",
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

st.divider()

# Prioritized customer table

st.subheader(
    "🚨 Prioritized Customer Signals"
)

display_columns = [
    "customer_id",
    "risk_level",
    "structured_risk_score",
    "plan",
    "region",
    "usage_change_pct",
    "avg_csat",
    "billing_issues_90d",
    "late_payments_90d",
    "unresolved_interactions",
]

available_columns = [
    c
    for c in display_columns
    if c in customers.columns
]
table = (
    customers
    .sort_values(
        "structured_risk_score",
        ascending=False
    )[available_columns]
    .copy()
)

st.dataframe(
    table.head(50),
    width="stretch",
    hide_index=True,
)

# Customer investigation

st.divider()

st.subheader(
    "🔍 Investigate Customer"
)

customer_id = st.selectbox(
    "Select a customer",
    customers["customer_id"].tolist()
)

customer = customers[
    customers["customer_id"] == customer_id
].iloc[0]

st.markdown(
    f"### {customer_id}"
)

# Customer metrics

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Structured Risk",
    f"{customer['structured_risk_score']:.0f}/100"
)

c2.metric(
    "Risk Level",
    customer["risk_level"]
)

c3.metric(
    "Usage Change",
    f"{customer['usage_change_pct']:.1f}%"
)

c4.metric(
    "CSAT",
    f"{customer['avg_csat']:.1f}/10"
)

# Risk drivers

st.markdown(
    "### ⚠️ Detected Risk Drivers"
)

risk_drivers = customer.get(
    "risk_drivers",
    []
)

if risk_drivers:
    for driver in risk_drivers:
        st.markdown(
            f"- {driver}"
        )
else:

    st.success(
        "No significant structured risk drivers detected."
    )

# Structured signals

st.markdown(
    "### 📊 Structured Signals"
)

signal_table = pd.DataFrame({
    "Signal": [
        "Usage change",
        "Average CSAT",
        "Billing issues",
        "Late payments",
        "Support interactions",
        "Unresolved interactions",
        "Feature adoption",
    ],
    "Value": [
        f"{customer['usage_change_pct']:.1f}%",
        f"{customer['avg_csat']:.1f}/10",
        str(
            int(
                customer["billing_issues_90d"]
            )
        ),
        str(
            int(
                customer["late_payments_90d"]
            )
        ),
        str(
            int(
                customer["support_interactions_90d"]
            )
        ),
        str(
            int(
                customer["unresolved_interactions"]
            )
        ),
        f"{customer['latest_feature_adoption_pct']:.1f}%",
    ],
})

st.dataframe(
    signal_table,
    width="stretch",
    hide_index=True,
)

# AI analysis

st.markdown(
    "### 🤖 AI Customer Signal Analysis"
)

st.caption(
    "Gemini correlates structured customer signals "
    "with recent support conversations."
)


if st.button(
    "Analyze Customer with AI",
    type="primary",
):

    transcripts = get_customer_transcripts(
        data["support"],
        customer_id,
        max_interactions=5,
    )

    if not transcripts:
        st.warning(
            "No support interactions found."
        )
    else:
        with st.spinner(
            "Correlating customer signals with "
            "support conversations..."
        ):

            try:
                analysis = analyze_customer(
                    customer.to_dict(),
                    transcripts,
                )

                st.session_state[
                    f"analysis_{customer_id}"
                ] = analysis

            except Exception as e:
                st.error(
                    f"AI analysis failed: {e}"
                )

# Display AI analysis

analysis = st.session_state.get(
    f"analysis_{customer_id}"
)

if analysis:
    st.divider()
    st.markdown(
        "### 🧠 AI Assessment"
    )

    a1, a2, a3, a4, a5 = st.columns(5)

    a1.metric(
        "AI Risk",
        f"{analysis.ai_risk_score:.0f}/100"
    )

    a2.metric(
        "Sentiment",
        analysis.sentiment.title()
    )

    a3.metric(
        "Frustration",
        f"{analysis.frustration_score:.0%}"
    )

    a4.metric(
        "Cancellation Intent",
        f"{analysis.cancellation_intent_score:.0%}"
    )

    a5.metric(
        "Urgency",
        f"{analysis.urgency_score:.0%}"
    )

    st.markdown(
        f"**Issue Category:** "
        f"{analysis.issue_category}"
    )

    # Signal correlations

    st.markdown(
        "#### 🔗 Signal Correlations"
    )

    for correlation in (
        analysis.signal_correlations
    ):
        st.markdown(
            f"- {correlation}"
        )

    # Key Signals
    st.markdown(
        "#### 🚩 Key Signals"
    )

    for signal in analysis.key_signals:
        st.markdown(
            f"- {signal}"
        )

    # Rationale

    st.markdown(
        "#### 💡 AI Rationale"
    )

    st.info(
        analysis.rationale
    )

    # Recommended action

    st.markdown(
        "#### 🎯 Recommended Action"
    )

    if analysis.ai_risk_score >= 60:
        st.error(
            analysis.recommended_action
        )

    elif analysis.ai_risk_score >= 30:
        st.warning(
            analysis.recommended_action
        )

    else:
        st.success(
            analysis.recommended_action
        )


    st.caption(
        f"AI confidence: "
        f"{analysis.confidence:.0%}"
    )

# Recent support interactions
with st.expander(
    "View Recent Support Interactions"
):

    interactions = data["support"][
        data["support"]["customer_id"]
        == customer_id
    ].sort_values(
        "timestamp",
        ascending=False
    ).head(5)

    for _, interaction in (
        interactions.iterrows()
    ):

        st.markdown(
            f"**{interaction['timestamp']} — "
            f"{interaction['issue_category']} — "
            f"{interaction['resolution_status']}**"
        )

        st.text(
            interaction["transcript"]
        )

        st.divider()