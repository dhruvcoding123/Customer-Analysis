import pandas as pd

def clamp(value, minimum=0, maximum=100):
    return max(minimum, min(maximum, value))


def calculate_structured_risk(row):
    """
    Calculate a 0-100 structured customer risk score.

    The score combines behavioral, support, satisfaction,
    and billing signals. Additional points are awarded when
    multiple negative signals occur together.
    """

    score = 0

    usage_change = float(
        row.get("usage_change_pct", 0) or 0
    )

    csat = float(
        row.get("avg_csat", 0) or 0
    )

    unresolved = int(
        row.get("unresolved_interactions", 0) or 0
    )

    billing_issues = int(
        row.get("billing_issues_90d", 0) or 0
    )

    late_payments = int(
        row.get("late_payments_90d", 0) or 0
    )

    support_interactions = int(
        row.get("support_interactions_90d", 0) or 0
    )

    feature_adoption = float(
        row.get("latest_feature_adoption_pct", 0) or 0
    )

    # 1. Usage deterioration: maximum 25 points
    
    if usage_change <= -50:
        score += 25
    elif usage_change <= -35:
        score += 21
    elif usage_change <= -20:
        score += 16
    elif usage_change <= -10:
        score += 9
    elif usage_change < 0:
        score += 4

    # 2. CSAT: maximum 20 points
    # Assumes CSAT is on a 1-10 scale.
    
    if csat > 0:

        if csat <= 3:
            score += 20
        elif csat <= 4:
            score += 17
        elif csat <= 5:
            score += 13
        elif csat <= 6:
            score += 9
        elif csat <= 7:
            score += 5
        elif csat <= 8:
            score += 2

    # 3. Unresolved support issues: maximum 15 points

    if unresolved >= 5:
        score += 15
    elif unresolved >= 4:
        score += 13
    elif unresolved >= 2:
        score += 9
    elif unresolved >= 1:
        score += 5

    # 4. Billing problems: maximum 15 points

    if billing_issues >= 4:
        score += 15
    elif billing_issues >= 3:
        score += 12
    elif billing_issues >= 2:
        score += 9
    elif billing_issues >= 1:
        score += 5

    # 5. Late payments: maximum 10 points

    if late_payments >= 3:
        score += 10
    elif late_payments >= 2:
        score += 7
    elif late_payments >= 1:
        score += 4

    # 6. Support volume: maximum 5 points

    if support_interactions >= 8:
        score += 5
    elif support_interactions >= 5:
        score += 3
    elif support_interactions >= 3:
        score += 1

    # 7. Feature adoption: maximum 5 points

    if feature_adoption > 0:

        if feature_adoption < 20:
            score += 5
        elif feature_adoption < 40:
            score += 3
        elif feature_adoption < 60:
            score += 1

    
    # 8. Multi-signal correlation bonus: maximum 5 points
    
    correlation_bonus = 0

    # Declining usage + unresolved support
    if usage_change <= -25 and unresolved >= 2:
        correlation_bonus += 2

    # Declining usage + low satisfaction
    if usage_change <= -25 and csat <= 5:
        correlation_bonus += 2

    # Billing friction + payment problems
    if billing_issues >= 2 and late_payments >= 1:
        correlation_bonus += 1

    score += min(correlation_bonus, 5)

    return clamp(score)


def get_risk_level(score):
    """
    Convert numeric risk score into business-friendly category.
    """

    if score >= 70:
        return "High"

    if score >= 40:
        return "Medium"

    return "Low"


def get_risk_drivers(row):
    """
    Return the major structured signals contributing
    to customer risk.
    """

    drivers = []

    usage_change = float(
        row.get("usage_change_pct", 0) or 0
    )

    csat = float(
        row.get("avg_csat", 0) or 0
    )

    unresolved = int(
        row.get("unresolved_interactions", 0) or 0
    )

    billing = int(
        row.get("billing_issues_90d", 0) or 0
    )

    late = int(
        row.get("late_payments_90d", 0) or 0
    )

    support = int(
        row.get("support_interactions_90d", 0) or 0
    )

    if usage_change <= -25:
        drivers.append(
            f"Usage declined {abs(usage_change):.1f}%"
        )

    if csat <= 5:
        drivers.append(
            f"Low CSAT ({csat:.1f}/10)"
        )

    if unresolved >= 2:
        drivers.append(
            f"{unresolved} unresolved support issues"
        )

    if billing >= 2:
        drivers.append(
            f"{billing} billing issues"
        )

    if late >= 1:
        drivers.append(
            f"{late} late payment(s)"
        )

    if support >= 5:
        drivers.append(
            f"High support contact volume ({support})"
        )

    if (
        usage_change <= -25
        and unresolved >= 2
    ):
        drivers.append(
            "Usage decline + unresolved support"
        )

    if (
        usage_change <= -25
        and csat <= 5
    ):
        drivers.append(
            "Usage decline + low CSAT"
        )

    if (
        billing >= 2
        and late >= 1
    ):
        drivers.append(
            "Billing issues + late payments"
        )

    return drivers


def add_structured_risk_score(df):
    df = df.copy()

    df["structured_risk_score"] = df.apply(
        calculate_structured_risk,
        axis=1
    )

    df["risk_level"] = df[
        "structured_risk_score"
    ].apply(get_risk_level)

    df["risk_drivers"] = df.apply(
        get_risk_drivers,
        axis=1
    )

    return df