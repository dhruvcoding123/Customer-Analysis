from typing import List

from google import genai
from pydantic import BaseModel, Field

from config import GEMINI_API_KEY, GEMINI_MODEL

client = genai.Client(
    api_key=GEMINI_API_KEY
)

class CustomerAnalysis(BaseModel):

    sentiment: str = Field(
        description=(
            "Overall customer sentiment: "
            "positive, neutral, or negative."
        )
    )

    frustration_score: float = Field(
        description=(
            "Customer frustration level from 0 to 1."
        )
    )

    cancellation_intent_score: float = Field(
        description=(
            "Likelihood of cancellation or churn intent "
            "from 0 to 1."
        )
    )

    urgency_score: float = Field(
        description=(
            "Urgency of intervention from 0 to 1."
        )
    )

    ai_risk_score: float = Field(
        description=(
            "AI-assessed customer risk from 0 to 100, "
            "considering both structured signals and "
            "support conversations."
        )
    )

    issue_category: str = Field(
        description=(
            "Primary customer issue category."
        )
    )

    signal_correlations: List[str] = Field(
        description=(
            "Important relationships between multiple "
            "customer signals."
        )
    )

    key_signals: List[str] = Field(
        description=(
            "Important early warning signals discovered "
            "from the structured data and conversations."
        )
    )

    rationale: str = Field(
        description=(
            "Concise business explanation of why the "
            "customer is or is not at risk."
        )
    )

    recommended_action: str = Field(
        description=(
            "One practical action for the customer "
            "operations or retention team."
        )
    )

    confidence: float = Field(
        description=(
            "Confidence in the overall assessment from 0 to 1."
        )
    )


def analyze_customer(
    customer,
    transcripts
):

    transcript_text = "\n\n".join(
        [
            f"Interaction {i + 1}:\n{text}"
            for i, text in enumerate(transcripts)
        ]
    )

    customer_id = customer.get(
        "customer_id",
        "Unknown"
    )

    prompt = f"""
You are an AI customer operations analyst.

Your job is to identify early warning signals of
customer dissatisfaction, churn, cancellation, or
escalation.

You must analyze BOTH structured customer signals
AND unstructured support conversations.

Do not analyze the transcript in isolation.

==================================================
CUSTOMER PROFILE
==================================================

Customer ID:
{customer_id}

Plan:
{customer.get("plan", "Unknown")}

Region:
{customer.get("region", "Unknown")}


==================================================
STRUCTURED SIGNALS
==================================================

Usage change:
{customer.get("usage_change_pct", 0)}%

Latest usage:
{customer.get("latest_usage_hours", 0)} hours

Latest sessions:
{customer.get("latest_sessions", 0)}

Feature adoption:
{customer.get("latest_feature_adoption_pct", 0)}%

Average CSAT:
{customer.get("avg_csat", 0)}/10

Latest CSAT:
{customer.get("latest_csat", 0)}/10

Billing issues in last 90 days:
{customer.get("billing_issues_90d", 0)}

Late payments in last 90 days:
{customer.get("late_payments_90d", 0)}

Refund requests in last 90 days:
{customer.get("refund_requests_90d", 0)}

Support interactions in last 90 days:
{customer.get("support_interactions_90d", 0)}

Unresolved support interactions:
{customer.get("unresolved_interactions", 0)}

Existing structured risk score:
{customer.get("structured_risk_score", 0)}/100

Existing risk level:
{customer.get("risk_level", "Unknown")}


==================================================
SUPPORT CONVERSATIONS
==================================================

{transcript_text}


==================================================
ANALYSIS REQUIREMENTS
==================================================

Evaluate:

1. Overall sentiment
2. Frustration
3. Cancellation/churn intent
4. Urgency
5. AI risk score from 0-100
6. Primary issue category
7. Important signal correlations
8. Key warning signals
9. Business rationale
10. Recommended action
11. Confidence

IMPORTANT:

- Correlate multiple signals rather than treating
  each signal independently.
- A large usage decline alone does not necessarily
  mean the customer will churn.
- A negative support message alone does not necessarily
  mean the customer is high risk.
- Look for combinations such as:
    * declining usage + low CSAT
    * declining usage + unresolved issues
    * billing issues + late payments
    * repeated support contacts + unresolved problems
    * negative sentiment + cancellation language
    * competitor mentions + declining engagement
- Consider whether reported problems were resolved.
- Explicit cancellation or competitor language is a
  strong churn indicator.
- A satisfied customer with resolved support interactions
  should not be classified as high risk merely because
  they contacted support.
- Do not invent facts that are not present in the data.
- Base the assessment only on the supplied information.
- Keep the rationale concise and useful to a customer
  operations team.

RISK GUIDANCE:

0-29   = Low
30-59  = Medium
60-79  = High
80-100 = Critical

Recommended actions must match the actual risk level.

For low-risk customers, recommend monitoring or
self-service education rather than unnecessary
retention intervention.

For high or critical-risk customers, recommend
specific proactive intervention.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema":
                CustomerAnalysis.model_json_schema(),
        },
    )

    return CustomerAnalysis.model_validate_json(
        response.text
    )


# Backward-compatible wrapper
def analyze_customer_transcripts(
    customer_id,
    transcripts
):

    customer = {
        "customer_id": customer_id
    }

    return analyze_customer(
        customer,
        transcripts
    )