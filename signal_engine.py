import pandas as pd
import numpy as np


def calculate_usage_signals(usage):
    usage = usage.copy()

    usage["month"] = pd.to_datetime(usage["month"])

    usage = usage.sort_values(
        ["customer_id", "month"]
    )

    results = []

    for customer_id, group in usage.groupby("customer_id"):

        group = group.sort_values("month")

        if len(group) < 2:
            continue

        first_usage = group.iloc[0]["usage_hours"]
        latest_usage = group.iloc[-1]["usage_hours"]

        if first_usage > 0:
            usage_change = (
                (latest_usage - first_usage)
                / first_usage
            ) * 100
        else:
            usage_change = 0

        latest = group.iloc[-1]

        results.append({
            "customer_id": customer_id,
            "usage_change_pct": round(usage_change, 2),
            "latest_usage_hours": round(
                latest["usage_hours"], 2
            ),
            "latest_sessions": int(
                latest["sessions"]
            ),
            "latest_feature_adoption_pct": round(
                latest["feature_adoption_pct"], 2
            ),
        })

    return pd.DataFrame(results)


def calculate_billing_signals(billing):
    billing = billing.copy()

    billing["invoice_date"] = pd.to_datetime(
        billing["invoice_date"]
    )

    cutoff = billing["invoice_date"].max() - pd.Timedelta(
        days=90
    )

    recent = billing[
        billing["invoice_date"] >= cutoff
    ]

    result = recent.groupby("customer_id").agg(
        late_payments_90d=(
            "payment_status",
            lambda x: int((x == "Late").sum())
        ),
        billing_issues_90d=(
            "billing_issue",
            "sum"
        ),
        refund_requests_90d=(
            "refund_requested",
            "sum"
        ),
        avg_invoice_amount=(
            "amount",
            "mean"
        ),
    ).reset_index()

    return result


def calculate_support_signals(support):
    support = support.copy()

    support["timestamp"] = pd.to_datetime(
        support["timestamp"]
    )

    cutoff = support["timestamp"].max() - pd.Timedelta(
        days=90
    )

    recent = support[
        support["timestamp"] >= cutoff
    ]

    result = recent.groupby("customer_id").agg(
        support_interactions_90d=(
            "interaction_id",
            "count"
        ),
        unresolved_interactions=(
            "resolution_status",
            lambda x: int(
                (x == "Unresolved").sum()
            )
        ),
        latest_interaction=(
            "timestamp",
            "max"
        ),
    ).reset_index()

    return result


def calculate_csat_signals(csat):

    csat = csat.copy()

    csat["feedback_date"] = pd.to_datetime(
        csat["feedback_date"]
    )

    cutoff = (
        csat["feedback_date"].max()
        - pd.Timedelta(days=90)
    )

    recent = csat[
        csat["feedback_date"] >= cutoff
    ].copy()

    recent = recent.sort_values(
        ["customer_id", "feedback_date"]
    )

    result = recent.groupby(
        "customer_id"
    ).agg(
        avg_csat=(
            "csat_score",
            "mean"
        ),
        latest_csat=(
            "csat_score",
            "last"
        ),
    ).reset_index()

    return result


def build_customer_signals(data):

    customers = data["customers"].copy()

    usage_signals = calculate_usage_signals(
        data["usage"]
    )

    billing_signals = calculate_billing_signals(
        data["billing"]
    )

    support_signals = calculate_support_signals(
        data["support"]
    )

    csat_signals = calculate_csat_signals(
        data["csat"]
    )

    customer_signals = customers.merge(
        usage_signals,
        on="customer_id",
        how="left"
    )

    customer_signals = customer_signals.merge(
        billing_signals,
        on="customer_id",
        how="left"
    )

    customer_signals = customer_signals.merge(
        support_signals,
        on="customer_id",
        how="left"
    )

    customer_signals = customer_signals.merge(
        csat_signals,
        on="customer_id",
        how="left"
    )

    numeric_columns = [
        "usage_change_pct",
        "late_payments_90d",
        "billing_issues_90d",
        "refund_requests_90d",
        "support_interactions_90d",
        "unresolved_interactions",
        "avg_csat",
        "latest_csat",
    ]

    for column in numeric_columns:
        if column in customer_signals.columns:
            customer_signals[column] = (
                customer_signals[column]
                .fillna(0)
            )

    return customer_signals

def get_customer_transcripts(
    support,
    customer_id,
    max_interactions=5
):

    customer_support = support[
        support["customer_id"] == customer_id
    ].copy()

    customer_support["timestamp"] = pd.to_datetime(
        customer_support["timestamp"]
    )

    customer_support = customer_support.sort_values(
        "timestamp",
        ascending=False
    )

    customer_support = customer_support.head(
        max_interactions
    )

    return customer_support[
        "transcript"
    ].tolist()